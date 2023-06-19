"""Basic PyTorch layers used for multiple Generative Models.
"""

import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True), pooling=False, use_bn=True, drop_ratio=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation_fun = activation
        self.dropout = nn.Dropout2d(p=drop_ratio)

        self.bn = None
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self.pool = None
        if pooling:
            self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        out = self.pool(x) if self.pool else x
        out = self.conv(out)
        out = self.bn(out) if self.bn else out
        out = self.activation_fun(out)
        out = self.dropout(out)
        return out


class BNReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True), pooling=False, use_bn=True, drop_ratio=0):
        super(BNReLUConv, self).__init__()
        self.activation_fun = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=drop_ratio)

        self.bn = None
        if use_bn:
            self.bn = nn.BatchNorm2d(in_channels)

        self.pool = None
        if pooling:
            self.pool = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        out = self.bn(x) if self.bn else x
        out = self.activation_fun(out)
        out = self.dropout(out)
        out = self.pool(out) if self.pool else out
        out = self.conv(out)
        return out
