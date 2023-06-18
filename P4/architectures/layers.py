"""Basic PyTorch layers used for multiple Generative Models.
"""

import torch.nn as nn


# Convolution + BatchNormnalization + ReLU block for the encoder

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.pool = None
        if pooling:
            self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        if (self.pool):
            out = self.pool(x)
        else:
            out = x
        out = self.relu(self.bn(self.conv(out)))
        return out


#  BatchNormnalization + ReLU block + Convolution for the decoder

class BNReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=False):
        super(BNReLUConv, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1)

        self.pool = None
        if pooling:
            self.pool = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        out = self.relu(self.bn(x))
        if (self.pool):
            out = self.pool(out)
        out = self.conv(out)
        return out