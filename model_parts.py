import torch
import torch.nn as nn

class ConvReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding='same'),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class RecurrentBlock(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super().__init__()
        
        self.conv1 = ConvReLU(in_channel, in_channel, kernel_size)
        self.conv2 = nn.Sequential(
            ConvReLU(2*in_channel, in_channel, kernel_size),
            ConvReLU(in_channel, in_channel, kernel_size)
        )

        self.h = None

    def forward(self, x):
        x = self.conv1(x)
        if self.h is None:
            self.h = torch.zeros_like(x, requires_grad=False)
        x = torch.cat((x, self.h), dim=1)
        x = self.conv2(x)
        self.h = x
        return x, self.h
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size):
        super().__init__()
        
        self.conv = nn.Sequential(
            ConvReLU(in_channel, mid_channel, kernel_size),
            ConvReLU(mid_channel, out_channel, kernel_size)
        )

    def forward(self, x, s):
        x = torch.cat((x, s), dim=1)
        x = self.conv(x)
        return x