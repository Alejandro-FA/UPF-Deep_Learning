"""Definition for a Variational Auto-Encoder model.
"""

import torch.nn as nn
from .GenerativeModel import *
from .layers import ConvBNReLU, BNReLUConv


# Encoder definition with 3 COnv-BN-ReLU blocks and fully-connected layer
class VAEEncoder(nn.Module):
    def __init__(self, out_features, base_channels=16):
        super(VAEEncoder, self).__init__()
        
        self.layer1 = ConvBNReLU(1, base_channels, pooling=False)
        self.layer2 = ConvBNReLU(base_channels, base_channels * 2, pooling=True)
        self.layer3 = ConvBNReLU(base_channels * 2, base_channels * 4, pooling=True)
        self.fc = nn.Linear(8 * 8 * base_channels * 4, out_features)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return self.fc(out.view(x.shape[0], -1))


# Decoder definition with a fully-connected layer and 3 BN-ReLU-COnv blocks and

class VAEDecoder(nn.ModuleDict):
    def __init__(self, out_features, base_channels=16):
        super(VAEDecoder, self).__init__()
        self.base_channels = base_channels
        self.fc = nn.Linear(out_features, 8 * 8 * base_channels * 4)
        self.layer3 = BNReLUConv(base_channels * 4, base_channels * 2, pooling=True)
        self.layer2 = BNReLUConv(base_channels * 2, base_channels, pooling=True)
        self.layer1 = BNReLUConv(base_channels, 1, pooling=False)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(x.shape[0], self.base_channels * 4, 8, 8)
        out = self.layer3(out)
        out = self.layer2(out)
        out = self.layer1(out)
        return torch.sigmoid(out)


class VAE(nn.Module, GenerativeModel):
    def __init__(self, out_features=32, base_channels=16):
        super(VAE, self).__init__()
        # Initialize the encoder and decoder using a dimensionality out_features for the vector z
        self.out_features = out_features
        self.encoder = VAEEncoder(out_features * 2, base_channels)
        self.decoder = VAEDecoder(out_features, base_channels)

    # function to obtain the mu and sigma of z for a samples x
    def encode(self, x):
        aux = self.encoder(x)
        # get z mean
        z_mean = aux[:, 0:self.out_features]
        # get z variance
        z_log_var = aux[:, self.out_features::]
        return z_mean, z_log_var

    # function to generate a random sample z given mu and sigma
    def sample_z(self, z_mean, z_log_var):  # NOTE: Reparametrization trick
        z_std = z_log_var.mul(0.5).exp()
        samples_unit_normal = torch.randn_like(z_mean)
        samples_z = samples_unit_normal*z_std + z_mean
        return samples_z

    # (1) encode a sample
    # (2) obtain a random vector z from mu and sigma
    # (3) Reconstruct the image using the decoder
    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        samples_z = self.sample_z(z_mean, z_log_var)
        x_rec = self.decoder(samples_z)
        return x_rec, z_mean, z_log_var
    
    
    def get_latent_space(self, n_samples, device='cpu'):
        z = torch.randn((n_samples, self.out_features)).to(device)
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    @property
    def name(self):
        return 'VAE'