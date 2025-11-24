import torch
import torch.nn as nn
from .separable_conv import SeparableConv2d

class SeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableBlock, self).__init__()
        self.sep_conv = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.sep_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
