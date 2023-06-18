"""Definition for a Variational Auto-Encoder model.
"""

from .GenerativeModel import *
from .modules import *


class VAE(nn.Module, GenerativeModel):
    def __init__(self, out_features=32, base_channels=16):
        super(VAE, self).__init__()
        # Initialize the encoder and decoder using a dimensionality out_features for the vector z
        self.out_features = out_features
        self.encoder = Encoder(out_features * 2, base_channels)
        self.decoder = Decoder(out_features, base_channels)

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