"""Definition for a Generative Adversarial Network model.
"""

import torch.nn as nn
from .modules import *
from .GenerativeModel import *


# Discriminator similar to VAE encoder
class Discriminator(nn.Module):
    def __init__(self, base_channels=16):
        super(Discriminator, self).__init__()
        # last fully connected layer acts as a a binary classifier
        self.classifier = Encoder(1,base_channels)

    # Forward pass obtaining the discriminator probability
    def forward(self,x):
        out = self.classifier(x)
        # use sigmoid to get the real/fake image probability
        return torch.sigmoid(out)

# Generator is defined as VAE decoder
class Generator(nn.Module):
    def __init__(self, in_features, base_channels=16):
        super(Generator, self).__init__()
        self.base_channels = base_channels
        self.in_features = in_features
        self.decoder = Decoder(out_features=in_features, base_channels=base_channels)

    # Generate an image from vector z
    def forward(self,z):
        return torch.sigmoid(self.decoder(z))

    # FIXME: remove if (nan)
    # # Sample a set of images from random vectors z
    # def sample(self,n_samples=256,device='cpu'):
    #     samples_unit_normal = torch.randn((n_samples,self.in_features)).to(device)
    #     return self.decoder(samples_unit_normal)
  

class GAN(nn.Module, GenerativeModel):
    def __init__(self, in_features=32, base_channels=16):
        super(GAN, self).__init__()
        self.discriminator = Discriminator(base_channels=base_channels)
        self.generator = Generator(in_features, base_channels=base_channels)

    def get_latent_space(self, n_samples, device='cpu'):
        return torch.randn((n_samples, self.generator.in_features)).to(device)
    
    def decode(self, z):
        return self.generator.decoder(z)
    
    @property
    def name(self):
        return "GAN"