import torch
import torch.nn as nn

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # output size = 1x1

    def forward(self, x):
        return self.pool(x)
