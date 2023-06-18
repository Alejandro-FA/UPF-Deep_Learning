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
            # input is ``(image_channels) x 64 x 64``
            nn.Conv2d(image_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
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

    def forward(self, input):
        return self.main(input)


# Generator is defined as VAE decoder
class Generator(nn.Module):
    def __init__(self, in_features, image_channels = 1, base_channels=32):
        super(Generator, self).__init__()
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
            nn.ConvTranspose2d( base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            # state size. ``(base_channels) x 32 x 32``
            nn.ConvTranspose2d( base_channels, image_channels, 4, 2, 1, bias=False),
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






























"""
TODO:
    - 
"""

# class Discriminator(nn.Module):
#     def __init__(self, out_features, base_channels=16):
#         super(Discriminator, self).__init__()
#         self.activation = nn.LeakyReLU(inplace=True)

#         self.layer1 = ConvBNReLU(
#             in_channels=1,
#             out_channels=base_channels,
#             activation=self.activation,
#             pooling=False,
#             use_bn=False
#         )
        
#         self.layer2 = ConvBNReLU(
#             in_channels=base_channels,
#             out_channels=base_channels * 2, 
#             activation=self.activation, 
#             pooling=True, 
#             use_bn=True
#         )
        
#         self.layer3 = ConvBNReLU(
#             in_channels=base_channels * 2,
#             out_channels=base_channels * 4,
#             activation=self.activation,
#             pooling=True,
#             use_bn=True
#         )
        
#         self.fc = nn.Linear(8 * 8 * base_channels * 4, out_features) # FIXME: remove

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.fc(out.view(x.shape[0], -1))
#         return torch.sigmoid(out)


# # Decoder definition with a fully-connected layer and 3 BN-ReLU-COnv blocks and

# class Generator(nn.Module):
#     def __init__(self, in_features, base_channels=16):
#         super(Generator, self).__init__()
#         self.base_channels = base_channels
#         self.activation = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(in_features, 8 * 8 * base_channels * 4) # FIXME: remove
#         self.layer3 = BNReLUConv(
#             base_channels * 4, 
#             base_channels * 2, 
#             activation=self.activation, 
#             pooling=True, 
#             use_bn=True
#         )

#         self.layer2 = BNReLUConv(
#             base_channels * 2, 
#             base_channels, 
#             activation=self.activation, 
#             pooling=True, 
#             use_bn=True
#         )
#       class Discriminator(nn.Module):
#     def __init__(self, out_features, base_channels=16):
#         super(Discriminator, self).__init__()
#         self.activation = nn.LeakyReLU(inplace=True)

#         self.layer1 = ConvBNReLU(
#             in_channels=1,
#             out_channels=base_channels,
#             activation=self.activation,
#             pooling=False,
#             use_bn=False
#         )
        
#         self.layer2 = ConvBNReLU(
#             in_channels=base_channels,
#             out_channels=base_channels * 2, 
#             activation=self.activation, 
#             pooling=True, 
#             use_bn=True
#         )
        
#         self.layer3 = ConvBNReLU(
#             in_channels=base_channels * 2,
#             out_channels=base_channels * 4,
#             activation=self.activation,
#             pooling=True,
#             use_bn=True
#         )
        
#         self.fc = nn.Linear(8 * 8 * base_channels * 4, out_features) # FIXME: remove

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.fc(out.view(x.shape[0], -1))
#         return torch.sigmoid(out)


# # Decoder definition with a fully-connected layer and 3 BN-ReLU-COnv blocks and

# class Generator(nn.Module):
#     def __init__(self, in_features, base_channels=16):
#         super(Generator, self).__init__()
#         self.base_channels = base_channels
#         self.activation = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(in_features, 8 * 8 * base_channels * 4) # FIXME: remove
#         self.layer3 = BNReLUConv(
#             base_channels * 4, 
#             base_channels * 2, 
#             activation=self.activation, 
#             pooling=True, 
#             use_bn=True
#         )

#         self.layer2 = BNReLUConv(
#             base_channels * 2, 
#             base_channels, 
#             activation=self.activation, 
#             pooling=True, 
#             use_bn=True
#         )
        
#         self.layer1 = BNReLUConv(
#             base_channels, 
#             1, 
#             activation=self.activation, 
#             pooling=False, 
#             use_bn=False
#         )      

#     def forward(self, x):
#         out = self.fc(x)
#         out = out.view(x.shape[0], self.base_channels * 4, 8, 8)
#         out = self.layer3(out)
#         out = self.layer2(out)
#         out = self.layer1(out)
#         return torch.tanh(out)



# class GAN(nn.Module, GenerativeModel):
#     def __init__(self, in_features=32, base_channels=16):
#         super(GAN, self).__init__()
#         self.in_features = in_features
#         self.discriminator = Discriminator(out_features=in_features, base_channels=base_channels)
#         self.generator = Generator(in_features, base_channels=base_channels)

#     def get_latent_space(self, n_samples, device='cpu'):
#         return torch.randn((n_samples, self.in_features)).to(device)
    
#     def decode(self, z):
#         return self.discriminator(z)
    
#     @property
#     def name(self):
#         return "GAN"


# class GAN(nn.Module, GenerativeModel):
#     def __init__(self, in_features=32, base_channels=16):
#         super(GAN, self).__init__()
#         self.in_features = in_features
#         self.discriminator = Discriminator(out_features=in_features, base_channels=base_channels)
#         self.generator = Generator(in_features, base_channels=base_channels)

#     def get_latent_space(self, n_samples, device='cpu'):
#         return torch.randn((n_samples, self.in_features)).to(device)
    
#     def decode(self, z):
#         return self.discriminator(z)
    
#     @property
#     def name(self):
#         return "GAN"