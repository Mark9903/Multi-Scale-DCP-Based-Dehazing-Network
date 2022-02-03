# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:51:09 2021

@author: wsfh9
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self, in_channels=3, depth_rate=8, kernel_size=3):
        super(MyNet, self).__init__()
        self.convin = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 3, dilation = 2)

    def forward(self, x):
        print(x.shape)
        x = self.convin(x)
        print(x.shape)

net = MyNet()
a = torch.randn(1, 3, 10, 10)
b = net(a)