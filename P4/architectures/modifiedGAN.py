"""Definition for a Generative Adversarial Network model.
"""

import torch
import torch.nn as nn
from .layers import ConvBNReLU, BNReLUConv
from .GenerativeModel import *


# Decoder definition with a fully-connected layer and 3 BN-ReLU-COnv blocks and


# Discriminator similar to VAE encoder
class Discriminator(nn.Module):
    def __init__(self, out_features=1, base_channels=32, image_channels=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # # input is ``(image_channels) x 64 x 64``
            # nn.Conv2d(image_channels, base_channels, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(image_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
    


# Generator is defined as VAE decoder
class Generator(nn.Module):
    def __init__(self, in_features, image_channels = 1, base_channels=32):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( in_features, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            # state size. ``(base_channels*8) x 4 x 4``
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            # state size. ``(base_channels*4) x 8 x 8``
            nn.ConvTranspose2d( base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            # state size. ``(base_channels*2) x 16 x 16``
            nn.ConvTranspose2d( base_channels * 2, image_channels, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(base_channels),
            # nn.ReLU(True),
            # # state size. ``(base_channels) x 32 x 32``
            # nn.ConvTranspose2d( base_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, x):
        return self.main(x)
  

class GAN(nn.Module, GenerativeModel):
    def __init__(self, in_features=32, base_channels=16):
        super(GAN, self).__init__()
        self.discriminator = Discriminator(base_channels=base_channels)
        self.generator = Generator(in_features, base_channels=base_channels)
        self.weights_init()

    def get_latent_space(self, n_samples, device='cpu'):
        return torch.randn(n_samples, self.generator.in_features, 1, 1).to(device)
    
    def decode(self, z):
        return self.generator(z)
    
    @property
    def name(self):
        return "GAN"
    
    def weights_init(self):
        """Initialize weights according to UNSUPERVISED REPRESENTATION LEARNING
        WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
        """
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(self.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(self.weight.data, 1.0, 0.02)
            nn.init.constant_(self.bias.data, 0)
