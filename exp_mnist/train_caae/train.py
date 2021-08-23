import warnings
warnings.filterwarnings("ignore")
import os, sys, configparser, logging, argparse
sys.path.append('/'.join(os.getcwd().split('/')[:-2]))
import torch
import torch.nn as nn

from XAE.model import CAAE_abstract
from XAE.logging_daily import logging_daily
from XAE.util import init_params

class CAAE_MNIST(CAAE_abstract):
    def __init__(self, cfg, log, device = 'cpu', verbose = 1):
        super(CAAE_MNIST, self).__init__(cfg, log, device, verbose)
        self.d = 1000
        d = self.d
        self.embed_data = nn.Identity().to(device)

        self.embed_condition = nn.Sequential(
            nn.Linear(self.y_dim, 28*28),
            nn.Unflatten(1, (1, 28, 28)),
            ).to(device)

        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*28*28, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),
            
            nn.Linear(d, self.z_dim),
            ).to(device)
     
        self.dec = nn.Sequential(
            nn.Linear(self.z_dim+self.y_dim, d),
            nn.ReLU(True),
            
            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),
            
            nn.Linear(d, 28*28),
            
            # reconstruction
            nn.Unflatten(1, (1, 28, 28)),
            nn.Tanh(),
            ).to(device)

        self.disc = nn.Sequential(
            nn.Linear(self.z_dim + self.y_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.Linear(16, 1),
            nn.Sigmoid(),
            ).to(device)
        
        init_params(self.enc)
        init_params(self.dec)
        init_params(self.disc)
        init_params(self.embed_condition)

        self.encoder_trainable = [self.enc]
        self.decoder_trainable = [self.dec]
        self.discriminator_trainable = [self.disc]
        self.embed_condition_trainable = [self.embed_condition]

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