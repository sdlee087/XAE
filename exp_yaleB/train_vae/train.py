import os, sys, configparser, logging, argparse
sys.path.append('/'.join(os.getcwd().split('/')[:-2]))
import torch
import torch.nn as nn

from XAE.model import VAE_abstract
from XAE.logging_daily import logging_daily
from XAE.util import init_params
    
class VAE_YaleB(VAE_abstract):
  def __init__(self, network_info, log, device = 'cpu', verbose = 1):
    super(VAE_YaleB, self).__init__(network_info, log, device, verbose)
    self.d = 64
    d = self.d
    self.enc = nn.Sequential(
        nn.Conv2d(1, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(True),

        nn.Conv2d(d, 2*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(True),

        nn.Conv2d(2*d, 4*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(True),

        nn.Conv2d(4*d, 8*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(True),
            
        nn.Flatten(),
        ).to(device)

    self.mu = nn.Linear(8*8*8*d, self.z_dim).to(device)
    self.logvar = nn.Linear(8*8*8*d, self.z_dim).to(device)
                                
    self.dec = nn.Sequential(
        nn.Linear(self.z_dim, 512, bias = False),
        nn.ReLU(True),
            
        nn.Linear(512, 16*16*8*d),
        nn.Unflatten(1, (8*d, 16, 16)),
            
        nn.ConvTranspose2d(8*d, 4*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(True),
            
        nn.ConvTranspose2d(4*d, 2*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(True),

        nn.ConvTranspose2d(2*d, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.ReLU(True),
            
        # reconstruction
        nn.Conv2d(d, 1, kernel_size = 3, padding = 1),
            
        ).to(device)
    init_params(self.enc)
    init_params(self.dec)
    init_params(self.mu)
    init_params(self.logvar)

    self.encoder_trainable = [self.enc, self.mu, self.logvar]
    self.decoder_trainable = [self.dec]

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    parser = argparse.ArgumentParser(description='Training Autoencoders')
    parser.add_argument('--log_info', type=str, help='path of file about log format')
    parser.add_argument('--train_config', type=str, help='path of training configuration file')
    args = parser.parse_args()

    logger = logging_daily(args.log_info)
    log = logger.get_logging()
    log.setLevel(logging.INFO)

    cfg = configparser.ConfigParser()
    cfg.read(args.train_config)

    network = getattr(sys.modules[__name__], cfg['train_info']['model_name'])
    model = network(cfg, log, device = device)
    model.train()