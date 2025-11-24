import torch
import torch.nn as nn
from .separable_block import SeparableBlock

class MiddleBlock(nn.Module):
    def __init__(self, channels, repetitions=8):
        super(MiddleBlock, self).__init__()
        self.repetitions = repetitions
        self.blocks = nn.ModuleList()
        for _ in range(repetitions):
            self.blocks.append(
                nn.Sequential(
                    SeparableBlock(channels, channels),
                    SeparableBlock(channels, channels),
                    SeparableBlock(channels, channels)
                )
            )

    def forward(self, x):
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual 
        return x
