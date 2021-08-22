import time
# sys.path.append('/'.join(os.getcwd().split('/')))
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

import numpy as np
import matplotlib.pyplot as plt

from .util import inc_avg, save_sample_images
from . import dataset, sampler

class AE_abstract(nn.Module):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(AE_abstract, self).__init__()
        self.log = log
        if verbose == 1:
            self.log.info('------------------------------------------------------------')
            for key in cfg['train_info']:
                self.log.info('%s : %s' % (key, cfg['train_info'][key]))

            for key in cfg['path_info']:
                self.log.info('%s : %s' % (key, cfg['path_info'][key]))

        self.cfg = cfg

        # Concrete Parts
        self.device = device
        self.z_sampler = getattr(sampler, cfg['train_info']['z_sampler']) # generate prior
        self.z_dim = int(cfg['train_info']['z_dim'])

        data_class = getattr(dataset, cfg['train_info']['train_data'])
        self.train_data =  data_class(cfg['path_info']['data_home'], train = True)
        self.test_data = data_class(cfg['path_info']['data_home'], train = False)
        self.prob_enc = cfg['train_info'].getboolean('prob_enc')

        self.batch_size = int(cfg['train_info']['batch_size'])
        self.validate_batch = cfg['train_info'].getboolean('validate')
        if cfg['train_info'].getboolean('replace'):
            it = int(cfg['train_info']['iter_per_epoch'])
            train_sampler = torch.utils.data.RandomSampler(self.train_data, replacement = True, num_samples = self.batch_size * it)
            self.train_generator = torch.utils.data.DataLoader(self.train_data, self.batch_size, num_workers = 5, sampler = train_sampler, pin_memory=True)
        else:
            self.train_generator = torch.utils.data.DataLoader(self.train_data, self.batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)

        self.test_generator = torch.utils.data.DataLoader(self.test_data, self.batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)
        
        self.save_best = cfg['train_info'].getboolean('save_best')
        self.save_path = cfg['path_info']['save_path']
        self.tensorboard_dir = cfg['path_info']['tb_logs']
        self.save_img_path = cfg['path_info']['save_img_path']
        self.save_state = cfg['path_info']['save_state']
        
        self.encoder_pretrain = cfg['train_info'].getboolean('encoder_pretrain')
        if self.encoder_pretrain:
            self.encoder_pretrain_batch_size = int(cfg['train_info']['encoder_pretrain_batch_size'])
            self.encoder_pretrain_step = int(cfg['train_info']['encoder_pretrain_max_step'])
            self.pretrain_generator = torch.utils.data.DataLoader(self.train_data, self.encoder_pretrain_batch_size, num_workers = 5, shuffle = True, pin_memory=True, drop_last=True)
        
        self.lr = float(cfg['train_info']['lr'])
        self.beta1 = float(cfg['train_info']['beta1'])
        self.lamb = float(cfg['train_info']['lambda'])
        self.lr_schedule = cfg['train_info']['lr_schedule']
        self.num_epoch = int(cfg['train_info']['epoch'])

        # Abstract Parts need overriding
        self.enc = nn.Identity()
        self.mu = nn.Identity()
        self.logvar = nn.Identity()
        self.dec = nn.Identity()

        self.recon_loss = nn.MSELoss()

        self.encoder_trainable = [self.enc]
        self.decoder_trainable = [self.dec]

    def reparameterize(self, mu, logvar):
        # reparameterization trick for probablistic encoder
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        if self.prob_enc:
            z = self.enc(x)
            return self.reparameterize(self.mu(z), self.logvar(z))
        else:
            return self.enc(x)
    
    def decode(self, z):
        return self.dec(z)

    def generate_prior(self, n):
        return self.z_sampler(n, self.z_dim, device = self.device)

    def generate_sample(self, n):
        return self.decode(self.z_sampler(n, self.z_dim, device = self.device))
        
    def forward(self, x):
        return self.decode(self.encode(x))

    def penalty_loss(self, x, y, n):
        # Overriding function for empirical distance between distributions
        raise NotImplementedError()

    def save(self, dir):
        torch.save(self.state_dict(), dir)

    def load(self, dir = None):
        if dir is None:
            self.load_state_dict(torch.load(self.save_path))
        self.load_state_dict(torch.load(dir))

    def lr_scheduler(self, optimizer, decay = 1.0):
        if self.lr_schedule is "manual":
            lamb = lambda e: decay * 1.0 * (0.5 ** (e >= 30)) * (0.2 ** (e >= 50)) * (0.1 ** (e >= 100))
            return optim.lr_scheduler.LambdaLR(optimizer, lamb)
        else:
            # Need update for basic scheduler
            lamb = lambda e: decay
            return optim.lr_scheduler.LambdaLR(optimizer, lamb)
        
    def pretrain_encoder(self):
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        mse = nn.MSELoss()
        
        self.log.info('------------------------------------------------------------')
        self.log.info('Pretraining Start!')
        
        cur_step = 0
        break_ind = False
        while True:
            for i, data in enumerate(self.pretrain_generator):
                cur_step = cur_step + 1
                pz = self.z_sampler(len(data), self.z_dim, device = self.device)
                x = data.to(self.device)
                qz = self.encode(x)

                qz_mean = torch.mean(qz, dim = 0)
                pz_mean = torch.mean(pz, dim = 0)

                qz_cov = torch.mean(torch.matmul((qz - qz_mean).unsqueeze(2), (qz - qz_mean).unsqueeze(1)), dim = 0)
                pz_cov = torch.mean(torch.matmul((pz - pz_mean).unsqueeze(2), (pz - pz_mean).unsqueeze(1)), dim = 0)

                loss = mse(pz_mean, qz_mean) + mse(pz_cov, qz_cov)

                loss.backward()
                optimizer.step()
                
                # train_loss_mse.append(loss.item(), len(data))
                if loss.item() > 0.1 or cur_step < 10:
                    print('train_mse: %.4f at %i step' % (loss.item(), cur_step), end = "\r")
                else:
                    self.log.info('train_mse: %.4f at %i step' % (loss.item(), cur_step))
                    break_ind = True
                    break
                    
            if break_ind or cur_step >= self.encoder_pretrain_step:
                break

    def train(self, resume = False):
        self.train_mse_list = []
        self.train_penalty_list = []
        self.test_mse_list = []
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
        # if self.lr_schedule is "manual":
        #     lamb = lambda e: 1.0 * (0.5 ** (e >= 30)) * (0.2 ** (e >= 50)) * (0.1 ** (e >= 100))
        #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lamb)

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
            train_loss_mse = inc_avg()
            train_loss_penalty = inc_avg()
            
            for i, data in enumerate(self.train_generator):

                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                n = len(data)

                prior_z = self.generate_prior(n)
                x = data.to(self.device)
                fake_latent = self.encode(x)
                recon = self.decode(fake_latent)
                
                loss = self.recon_loss(x, recon)
                if self.lamb > 0:
                    penalty = self.penalty_loss(fake_latent, prior_z, n)
                    obj = loss + self.lamb * penalty
                else:
                    obj = loss
                obj.backward()
                optimizer.step()
                
                train_loss_mse.append(loss.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n)
                
                print('[%i/%i]\ttrain_mse: %.4f\ttrain_penalty: %.4f' % (i+1, len(self.train_generator), train_loss_mse.avg, train_loss_penalty.avg), 
                      end = "\r")

            self.train_mse_list.append(train_loss_mse.avg)
            self.train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/MSE', train_loss_mse.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_mse = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, data in enumerate(self.test_generator):

                    n = len(data)

                    prior_z = self.generate_prior(n)
                    x = data.to(self.device)
                    fake_latent = self.encode(x)
                    recon = self.decode(fake_latent)

                    test_loss_mse.append(self.recon_loss(x, recon).item(), n)
                    if self.lamb > 0:
                        test_loss_penalty.append(self.penalty_loss(fake_latent, prior_z, self.test_generator.batch_size).item(), n)
                    print('[%i/%i]\ttest_mse: %.4f\ttest_penalty: %.4f' % (i, len(self.test_generator), test_loss_mse.avg, test_loss_penalty.avg), end = "\r")

                self.test_mse_list.append(test_loss_mse.avg)
                self.test_penalty_list.append(test_loss_penalty.avg)
                
                self.log.info('[%d/%d]\ttrain_mse: %.6e\ttrain_penalty: %.6e\ttest_mse: %.6e\ttest_penalty: %.6e'
                      % (epoch + 1, self.num_epoch, train_loss_mse.avg, train_loss_penalty.avg, test_loss_mse.avg, test_loss_penalty.avg))

                # Additional test set
                data = next(iter(self.test_generator))

                x = data.to(self.device)
                prior_z = self.generate_prior(len(data))
                fake_latent = self.encode(x).detach()
                recon = self.decode(fake_latent).detach()

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/MSE', test_loss_mse.avg, epoch)
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
                        test_dec = self.dec(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_mse.avg + self.lamb * test_loss_penalty.avg
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

class AE_adv_abstract(AE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(AE_adv_abstract, self).__init__(cfg, log, device, verbose)
        self.lr_adv = float(cfg['train_info']['lr_adv'])
        self.beta1_adv = float(cfg['train_info']['beta1_adv'])

        # Abstract Parts need overriding
        self.disc = nn.Identity()
        self.discriminator_trainable = [self.disc]

    def discriminate(self, z):
        return self.disc(z)

    def adv_loss(self, x, y):
        raise NotImplementedError()

    def penalty_loss(self, x):
        raise NotImplementedError()

    def train(self, resume = False):
        self.train_mse_list = []
        self.train_penalty_list = []
        self.test_mse_list = []
        self.test_penalty_list = []

        for net in self.encoder_trainable:
            net.train()
        for net in self.decoder_trainable:
            net.train()
        for net in self.discriminator_trainable:
            net.train()
            
        if self.encoder_pretrain:
            self.pretrain_encoder()
            self.log.info('Pretraining Ended!')
            
        if len(self.tensorboard_dir) > 0:
            self.writer = SummaryWriter(self.tensorboard_dir)
            
        optimizer_main = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        optimizer_adv = optim.Adam(sum([list(net.parameters()) for net in self.discriminator_trainable], []), lr = self.lr_adv, betas = (self.beta1_adv, 0.999))

        start_epoch = 0
        scheduler_main = self.lr_scheduler(optimizer_main)
        scheduler_adv = self.lr_scheduler(optimizer_adv)

        if resume:
            checkpoint = torch.load(self.save_state)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_main.load_state_dict(checkpoint['optimizer_main_state_dict'])
            optimizer_adv.load_state_dict(checkpoint['optimizer_adv_state_dict'])
            if len(self.lr_schedule) > 0:
                scheduler_main.load_state_dict(checkpoint['scheduler_main'])
                scheduler_adv.load_state_dict(checkpoint['scheduler_adv'])

        self.log.info('------------------------------------------------------------')
        self.log.info('Training Start!')
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epoch):
            # train_step
            train_loss_mse = inc_avg()
            train_loss_penalty = inc_avg()
            
            for i, data in enumerate(self.train_generator):
                n = len(data)
                prior_z = self.generate_prior(n)
                x = data.to(self.device)

                for net in self.encoder_trainable:
                    net.zero_grad()
                for net in self.decoder_trainable:
                    net.zero_grad()

                if self.lamb > 0:
                    for net in self.discriminator_trainable:
                        net.zero_grad()

                    fake_latent = self.encode(x)
                    obj_adv = self.lamb * self.adv_loss(prior_z, fake_latent)
                    obj_adv.backward()
                    optimizer_adv.step()

                fake_latent = self.encode(x)
                recon = self.decode(fake_latent)
                penalty = self.penalty_loss(fake_latent)
                
                loss = self.recon_loss(x, recon)
                if self.lamb > 0:
                    obj_main = loss + self.lamb * penalty
                else:
                    obj_main = loss
                obj_main.backward()
                optimizer_main.step()
                
                train_loss_mse.append(loss.item(), n)
                if self.lamb > 0:
                    train_loss_penalty.append(penalty.item(), n)
                
                print('[%i/%i]\ttrain_mse: %.4f\ttrain_penalty: %.4f' % (i+1, len(self.train_generator), train_loss_mse.avg, train_loss_penalty.avg), 
                      end = "\r")

            self.train_mse_list.append(train_loss_mse.avg)
            self.train_penalty_list.append(train_loss_penalty.avg)

            if len(self.tensorboard_dir) > 0:
                self.writer.add_scalar('train/MSE', train_loss_mse.avg, epoch)
                self.writer.add_scalar('train/penalty', train_loss_penalty.avg, epoch)
                if self.cfg['train_info'].getboolean('histogram'):
                    for param_tensor in self.state_dict():
                        self.writer.add_histogram(param_tensor, self.state_dict()[param_tensor].detach().to('cpu').numpy().flatten(), epoch)
            
            # validation_step
            test_loss_mse = inc_avg()
            test_loss_penalty = inc_avg()

            if self.validate_batch:
                for i, data in enumerate(self.test_generator):

                    n = len(data)

                    prior_z = self.generate_prior(n)
                    x = data.to(self.device)
                    fake_latent = self.encode(x)
                    recon = self.decode(fake_latent)

                    penalty = self.penalty_loss(fake_latent)

                    test_loss_mse.append(self.recon_loss(x, recon).item(), n)
                    if self.lamb > 0:
                        test_loss_penalty.append(penalty.item(), n)
                    print('[%i/%i]\ttest_mse: %.4f\ttest_penalty: %.4f' % (i, len(self.test_generator), test_loss_mse.avg, test_loss_penalty.avg), end = "\r")

                self.test_mse_list.append(test_loss_mse.avg)
                self.test_penalty_list.append(test_loss_penalty.avg)
                
                self.log.info('[%d/%d]\ttrain_mse: %.6e\ttrain_penalty: %.6e\ttest_mse: %.6e\ttest_penalty: %.6e'
                      % (epoch + 1, self.num_epoch, train_loss_mse.avg, train_loss_penalty.avg, test_loss_mse.avg, test_loss_penalty.avg))

                # Additional test set
                data = next(iter(self.test_generator))

                x = data.to(self.device)
                prior_z = self.generate_prior(len(data))
                fake_latent = self.encode(x).detach()
                recon = self.decode(fake_latent).detach()

                if len(self.tensorboard_dir) > 0:
                    self.writer.add_scalar('test/MSE', test_loss_mse.avg, epoch)
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
                        test_dec = self.dec(prior_z).detach().to('cpu').numpy()
                        save_sample_images('%s/gen' % self.save_img_path, epoch, (test_dec[0:64])*0.5 + 0.5)
                        plt.close()
                
                if self.save_best:
                    obj = test_loss_mse.avg + self.lamb * test_loss_penalty.avg
                    if self.best_obj[1] > obj:
                        self.best_obj[0] = epoch + 1
                        self.best_obj[1] = obj
                        self.save(self.save_path)
                        self.log.info("model saved, obj: %.6e" % obj)
                else:
                    self.save(self.save_path)
                    # self.log.info("model saved at: %s" % self.save_path)
                
            scheduler_main.step()
            if self.lamb > 0:
                scheduler_adv.step()

            if len(self.save_state) > 0:
                save_dict = {
                    'epoch':epoch + 1,
                    'model_state_dict':self.state_dict(),
                    'optimizer_main_state_dict':optimizer_main.state_dict(),
                    'optimizer_adv_state_dict':optimizer_adv.state_dict()
                } 
                if len(self.lr_schedule) > 0:
                    save_dict['scheduler_main'] = scheduler_main.state_dict()
                    save_dict['scheduler_adv'] = scheduler_adv.state_dict()
                torch.save(save_dict, self.save_state)
            
        if not self.validate_batch:
            self.save(self.save_path)

        self.log.info('Training Finished!')
        self.log.info("Elapsed time: %.3fs" % (time.time() - start_time))

        if len(self.tensorboard_dir) > 0:
            self.writer.close()