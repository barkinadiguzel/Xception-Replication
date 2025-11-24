import torch
import torch.nn as nn
from layers.xception_block import XceptionBlock
from layers.middle_block import MiddleBlock
from layers.pooling.global_avgpool import GlobalAvgPool
from layers.flatten_layer import FlattenLayer

class Xception(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3):
        super(Xception, self).__init__()
        
        # Entry Flow
        self.entry_conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=0, bias=False)
        self.entry_bn1 = nn.BatchNorm2d(32)
        self.entry_relu1 = nn.ReLU(inplace=True)
        
        self.entry_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.entry_bn2 = nn.BatchNorm2d(64)
        self.entry_relu2 = nn.ReLU(inplace=True)
        
        self.block1 = XceptionBlock(64, 128, reps=3, stride=2)
        self.block2 = XceptionBlock(128, 256, reps=3, stride=2)
        self.block3 = XceptionBlock(256, 728, reps=3, stride=2)

        # Middle Flow
        self.middle_flow = MiddleBlock(728, repetitions=8)
        
        # Exit Flow
        self.block_exit1 = XceptionBlock(728, 1024, reps=3, stride=2)
        self.sep_exit = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1536, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            nn.Conv2d(1536, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        
        # Pooling + Flatten
        self.global_pool = GlobalAvgPool()
        self.flatten = FlattenLayer()
        
        # Classifier
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Entry flow
        x = self.entry_conv1(x)
        x = self.entry_bn1(x)
        x = self.entry_relu1(x)
        
        x = self.entry_conv2(x)
        x = self.entry_bn2(x)
        x = self.entry_relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.middle_flow(x)
        
        # Exit flow
        x = self.block_exit1(x)
        x = self.sep_exit(x)
        
        # Pooling + Flatten
        x = self.global_pool(x)
        x = self.flatten(x)
        
        # Classifier
        x = self.fc(x)
        return x
