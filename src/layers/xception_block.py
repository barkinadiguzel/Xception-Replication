import torch
import torch.nn as nn
from .separable_block import SeparableBlock

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps=3, stride=1):
        super(XceptionBlock, self).__init__()
        self.reps = reps
        self.stride = stride

        layers = []
        for i in range(reps):
            s = stride if i == reps - 1 else 1  # only last block may have stride
            layers.append(SeparableBlock(in_channels if i == 0 else out_channels, out_channels, stride=s))
        self.blocks = nn.Sequential(*layers)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.residual = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.blocks(x)
        if self.residual is not None:
            residual = self.bn(self.residual(residual))
        x = x + residual
        x = self.relu(x)
        return x
