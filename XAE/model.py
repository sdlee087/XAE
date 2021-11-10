import time
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except ImportError:
    pass

from .util import inc_avg, save_sample_images, reparameterize
from . import dataset, sampler
from ._base_model import XAE_abstract, XAE_adv_abstract, CXAE_abstract, CXAE_adv_abstract

import numpy as np
import matplotlib.pyplot as plt

class WAE_MMD_abstract(XAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(WAE_MMD_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.MSELoss()

    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*self.z_dim*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat
        
    def penalty_loss(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

class CWAE_MMD_abstract(CXAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CWAE_MMD_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.MSELoss()

    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*self.z_dim*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat
        
    def penalty_loss(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

class WAE_GAN_abstract(XAE_adv_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(WAE_GAN_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.MSELoss()
        self.disc_loss = nn.BCEWithLogitsLoss()

    def penalty_loss(self, q):
        qz = self.discriminate(q)
        return self.disc_loss(qz, torch.ones_like(qz))

    def adv_loss(self, p, q):
        pz = self.discriminate(p)
        qz = self.discriminate(q)
        return self.disc_loss(pz, torch.ones_like(pz)) + self.disc_loss(qz, torch.zeros_like(qz))

class CWAE_GAN_abstract(CXAE_adv_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CWAE_GAN_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.MSELoss()
        self.disc_loss = nn.BCEWithLogitsLoss()

    def penalty_loss(self, q):
        qz = self.discriminate(q)
        return self.disc_loss(qz, torch.ones_like(qz))

    def adv_loss(self, p, q):
        pz = self.discriminate(p)
        qz = self.discriminate(q)
        return self.disc_loss(pz, torch.ones_like(pz)) + self.disc_loss(qz, torch.zeros_like(qz))
        
class VAE_abstract(XAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(VAE_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        # self.log2pi = torch.log(2.0 * np.pi)
        self.mu = nn.Identity()
        self.logvar = nn.Identity()

    def encode(self, x):
        xx = self.enc(x)
        return reparameterize(self.mu(xx), self.logvar(xx))

    def decode(self, x):
        return self.dec(x).tanh()

    def mu_and_logvar(self, x):
        xx = self.enc(x)
        return self.mu(xx), self.logvar(xx)
        
    def penalty_loss(self, mu, logvar):
        return (-.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim = 1)).mean()

    def pretrain_encoder(self):
        raise NotImplementedError()

    def train(self, resume = False):
        self.train_main_list = []
        self.train_penalty_list = []
        self.test_main_list = []
        self.test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))

        start_epoch = 0
        scheduler = self.lr_scheduler(optimizer)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler.load_state_dict(checkpoint['scheduler'])

        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_penalty = inc_avg()
            
            for i, data in enumerate(self.train_generator):

                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                n = len(data)
                x = data.to(self.device)
                mu, logvar = self.mu_and_logvar(x)
                recon = self.dec(reparameterize(mu, logvar))
                
                loss = self.main_loss((recon + 1.0)*0.5, (x + 1.0)*0.5) / n
                if self.lamb > 0:
                    penalty = self.penalty_loss(mu, logvar)
                    obj = loss + self.lamb * penalty
                else:
                    obj = loss
                obj.backward()
                optimizer.step()
                
                train_loss_main.append(loss.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n)
                
                print('[%i/%i]\ttrain_main: %.4f\ttrain_penalty: %.4f' % (i+1, len(self.train_generator), train_loss_main.avg, train_loss_penalty.avg), 
                      end = "\r")

            self.train_main_list.append(train_loss_main.avg)
            self.train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, data in enumerate(self.test_generator):

                    n = len(data)
                    x = data.to(self.device)
                    mu, logvar = self.mu_and_logvar(x)
                    recon = self.dec(reparameterize(mu, logvar))

                    test_loss_main.append(self.main_loss((recon + 1.0)*0.5, (x + 1.0)*0.5).item() / n, n)
                    if self.lamb > 0:
                        test_loss_penalty.append(self.penalty_loss(mu, logvar).item(), n)
                    print('[%i/%i]\ttest_main: %.4f\ttest_penalty: %.4f' % (i, len(self.test_generator), test_loss_main.avg, test_loss_penalty.avg), end = "\r")

                self.test_main_list.append(test_loss_main.avg)
                self.test_penalty_list.append(test_loss_penalty.avg)
                
                self.log.info('[%d/%d]\ttrain_main: %.6e\ttrain_penalty: %.6e\ttest_main: %.6e\ttest_penalty: %.6e'
                      % (epoch + 1, self.num_epoch, train_loss_main.avg, train_loss_penalty.avg, test_loss_main.avg, test_loss_penalty.avg))

                # Additional test set
                data = next(iter(self.test_generator))

                x = data.to(self.device)
                fake_latent = self.encode(x).detach()
                recon = self.decode(fake_latent).detach()
                prior_z = self.generate_prior(len(data)).detach()

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = fake_latent.to('cpu').numpy()
                        for_embed2 = prior_z.to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.to('cpu').numpy()[0:16], recon.to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.to('cpu').numpy()[0:32], recon.to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler'] = scheduler.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()

class CVAE_abstract(XAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CVAE_abstract, self).__init__(cfg, log, device, verbose)
        self.loss = nn.BCEWithLogitsLoss(reduction = 'sum')
        self.y_sampler = getattr(sampler, cfg['train_info']['y_sampler']) # generate prior
        self.y_dim = int(cfg['train_info']['y_dim'])

        self.mu = nn.Identity()
        self.logvar = nn.Identity()
        self.embed_data = nn.Identity()
        self.embed_condition = nn.Identity()

    def generate_prior(self, n):
        return self.z_sampler(n, self.z_dim, device = self.device), self.y_sampler(n, self.y_dim, device = self.device)

    def encode(self, x, y):
        xx = self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1))
        return reparameterize(self.mu(xx), self.logvar(xx))

    def decode(self, x, y):
        return self.dec(torch.cat((x, y), dim = 1)).tanh()

    def forward(self, x, y):
        return self.decode(self.encode(x,y), y)

    def mu_and_logvar(self, x, y):
        xx = self.enc(torch.cat((self.embed_data(x), self.embed_condition(y)), dim = 1))
        return self.mu(xx), self.logvar(xx)
        
    def penalty_loss(self, mu, logvar):
        return (-.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim = 1)).mean()

    def pretrain_encoder(self):
        raise NotImplementedError()

    def train(self, resume = False):
        self.train_main_list = []
        self.train_penalty_list = []
        self.test_main_list = []
        self.test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))

        start_epoch = 0
        scheduler = self.lr_scheduler(optimizer)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler.load_state_dict(checkpoint['scheduler'])

        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_main = inc_avg()
            train_loss_penalty = inc_avg()
            
            for i, (data, condition) in enumerate(self.train_generator):
                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                n = len(data)
                x = data.to(self.device)
                y = condition.to(self.device)
                mu, logvar = self.mu_and_logvar(x, y)
                recon = self.dec(torch.cat((reparameterize(mu, logvar), y), dim = 1))
                
                loss = self.main_loss((recon + 1.0)*0.5, (x + 1.0)*0.5) / n
                if self.lamb > 0:
                    penalty = self.penalty_loss(mu, logvar)
                    obj = loss + self.lamb * penalty
                else:
                    obj = loss
                obj.backward()
                optimizer.step()
                
                train_loss_main.append(loss.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n)
                
                print('[%i/%i]\ttrain_main: %.4f\ttrain_penalty: %.4f' % (i+1, len(self.train_generator), train_loss_main.avg, train_loss_penalty.avg), 
                      end = "\r")

            self.train_main_list.append(train_loss_main.avg)
            self.train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/main', train_loss_main.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_main = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, (data, condition) in enumerate(self.test_generator):

                    n = len(data)
                    x = data.to(self.device)
                    y = condition.to(self.device)
                    mu, logvar = self.mu_and_logvar(x, y)
                    recon = self.dec(torch.cat((reparameterize(mu, logvar), y), dim = 1))

                    test_loss_main.append(self.main_loss((recon + 1.0)*0.5, (x + 1.0)*0.5).item() / n, n)
                    if self.lamb > 0:
                        test_loss_penalty.append(self.penalty_loss(mu, logvar).item(), n)
                    print('[%i/%i]\ttest_main: %.4f\ttest_penalty: %.4f' % (i, len(self.test_generator), test_loss_main.avg, test_loss_penalty.avg), end = "\r")

                self.test_main_list.append(test_loss_main.avg)
                self.test_penalty_list.append(test_loss_penalty.avg)
                
                self.log.info('[%d/%d]\ttrain_main: %.6e\ttrain_penalty: %.6e\ttest_main: %.6e\ttest_penalty: %.6e'
                      % (epoch + 1, self.num_epoch, train_loss_main.avg, train_loss_penalty.avg, test_loss_main.avg, test_loss_penalty.avg))

                # Additional test set
                data, condition = next(iter(self.test_generator))

                x = data.to(self.device)
                y = condition.to(self.device)
                fake_latent = self.encode(x, y)
                recon = self.decode(fake_latent, y).detach()
                prior_z, prior_y = self.generate_prior(len(data))

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/main', test_loss_main.avg, epoch)
                    self.writer.add_scalar('test/penalty', test_loss_penalty.avg, epoch)

                    if self.lamb > 0:
                        # Embedding
                        for_embed1 = torch.cat((fake_latent, y), dim = 1).detach().to('cpu').numpy()
                        for_embed2 = torch.cat((prior_z, prior_y), dim = 1).detach().to('cpu').numpy()
                        label = ['fake']*len(for_embed1) + ['prior']*len(for_embed2)
                        self.writer.add_embedding(np.concatenate((for_embed1, for_embed2)), metadata = label, global_step = epoch)

                        # Sample Generation
                        test_dec = self.decode(prior_z, prior_y).detach().to('cpu').numpy()
                        self.writer.add_images('generated_sample', (test_dec[0:32])*0.5 + 0.5, epoch)

                    # Reconstruction
                    self.writer.add_images('reconstruction', (np.concatenate((x.to('cpu').numpy()[0:16], recon.to('cpu').numpy()[0:16])))*0.5 + 0.5, epoch)
                    self.writer.flush()

                if len(self.save_img_path) > 0:
                    save_sample_images('%s/recon' % self.save_img_path, epoch, (np.concatenate((x.to('cpu').numpy()[0:32], recon.to('cpu').numpy()[0:32])))*0.5 + 0.5)
                    plt.close()
                    if self.lamb > 0:
                        # Sample Generation
                        test_dec = self.decode(prior_z, prior_y).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_main.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler'] = scheduler.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()