"""Definition for a Generative Adversarial Network model.
"""

import torch
import torch.nn as nn
from .layers import ConvBNReLU, BNReLUConv
from .GenerativeModel import *


# Decoder definition with a fully-connected layer and 3 BN-ReLU-COnv blocks and



# Discriminator similar to VAE encoder
class Discriminator(nn.Module):
    def __init__(self, out_features=1, base_channels=16):
        super(Discriminator, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        self.layer1 = ConvBNReLU(
            1, 
            base_channels, 
            activation=self.activation, 
            pooling=False,
            use_bn=True,
            drop_ratio=0.5
        )
        
        self.layer2 = ConvBNReLU(
            base_channels, 
            base_channels * 2, 
            activation=self.activation, 
            pooling=True,
            use_bn=True,
            drop_ratio=0.5
        )
        
        self.layer3 = ConvBNReLU(
            base_channels * 2,
            base_channels * 4, 
            activation=self.activation, 
            pooling=True,
            use_bn=True,
            drop_ratio=0.5
        )
        
        self.fc = nn.Linear(8 * 8 * base_channels * 4, out_features)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out.view(x.shape[0], -1))
        return torch.sigmoid(out)


# Generator is defined as VAE decoder
class Generator(nn.Module):
    def __init__(self, in_features, base_channels=16):
        super(Generator, self).__init__()
        self.base_channels = base_channels
        self.in_features = in_features
        self.activation = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features, 8 * 8 * base_channels * 4)
        
        self.layer3 = BNReLUConv(
            base_channels * 4,
            base_channels * 2,
            activation=self.activation,
            pooling=True,
            use_bn=True,
            drop_ratio=0.5
        )
        
        self.layer2 = BNReLUConv(
            base_channels * 2, 
            base_channels, 
            activation=self.activation, 
            pooling=True,
            use_bn=True,
            drop_ratio=0.5
        )
        
        self.layer1 = BNReLUConv(
            base_channels,
            1,
            activation=self.activation,
            pooling=False,
            use_bn=True,
            drop_ratio=0.5
        ) 

    def forward(self, x):
        out = self.fc(x)
        out = out.view(x.shape[0], self.base_channels * 4, 8, 8)
        out = self.layer3(out)
        out = self.layer2(out)
        out = self.layer1(out)
        return torch.sigmoid(out) # TODO: change to tanh?
  

class GAN(nn.Module, GenerativeModel):
    def __init__(self, in_features=32, base_channels=16):
        super(GAN, self).__init__()
        self.discriminator = Discriminator(base_channels=base_channels)
        self.generator = Generator(in_features, base_channels=base_channels)
        self.weights_init()

    def get_latent_space(self, n_samples, device='cpu'):
        return torch.randn((n_samples, self.generator.in_features)).to(device)
    
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
