"""Multi-layer modules shared by multiple Generative Models.
"""

import torch
import torch.nn as nn
from .layers import *


# Encoder definition with 3 COnv-BN-ReLU blocks and fully-connected layer

class Encoder(nn.Module):
    def __init__(self, out_features, base_channels=16):
        super(Encoder, self).__init__()
        self.layer1 = ConvBNReLU(1, base_channels, pooling=False)
        self.layer2 = ConvBNReLU(base_channels, base_channels*2, pooling=True)
        self.layer3 = ConvBNReLU(
            base_channels*2, base_channels*4, pooling=True)
        self.fc = nn.Linear(8*8*base_channels*4, out_features)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return self.fc(out.view(x.shape[0], -1))


# Decoder definition with a fully-connected layer and 3 BN-ReLU-COnv blocks and

class Decoder(nn.Module):
    def __init__(self, out_features, base_channels=16):
        super(Decoder, self).__init__()
        self.base_channels = base_channels
        self.fc = nn.Linear(out_features, 8 * 8 * base_channels * 4)
        self.layer3 = BNReLUConv(
            base_channels*4, base_channels * 2, pooling=True)
        self.layer2 = BNReLUConv(base_channels * 2, base_channels, pooling=True)
        self.layer1 = BNReLUConv(base_channels, 1, pooling=False)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(x.shape[0], self.base_channels * 4, 8, 8)
        out = self.layer3(out)
        out = self.layer2(out)
        out = self.layer1(out)
        return torch.sigmoid(out)