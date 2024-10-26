import torch
import torch.nn as nn

class ConvSig(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding='same'),
            nn.LeakyReLU(negative_slope=0.01)
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
    
class EncoderBlock(nn.Module):
    """ ConvReLU -> ConvReLU -> =Skip -> Down"""
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size):
        super().__init__()
        
        self.down = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            ConvReLU(in_channel, mid_channel, kernel_size),
            ConvReLU(mid_channel, out_channel, kernel_size)
        )

    def forward(self, x):
        s = x = self.conv(x)
        x = self.down(x)
        return x, s
    
class DecoderBlock(nn.Module):
    """ +Skip -> ConvReLU -> ConvReLU -> Up """
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            ConvReLU(in_channel, mid_channel, kernel_size),
            ConvReLU(mid_channel, out_channel, kernel_size)
        )

    def forward(self, x, s):
        x = torch.cat((x, s), dim=1)
        x = self.conv(x)
        x = self.up(x)
        return x
    
class Bottleneck(nn.Module):
    """ ConvReLU -> ConvReLU -> Up"""
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            ConvReLU(in_channel, mid_channel, kernel_size),
            ConvReLU(mid_channel, out_channel, kernel_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x
    
class OutputBlock(nn.Module):
    """ ConvReLU -> ConvSig """
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size):
        super().__init__()
        
        self.conv = nn.Sequential(
            ConvReLU(in_channel, mid_channel, kernel_size),
            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=kernel_size, padding='same')
        )

        self.sig = nn.Sigmoid()

    def forward(self, x, s):
        """
        x : N x C x H x W
        s : N x C x H x W
        
        f : [0,24)  | ?
        a : [24,27) | Square
        c : [27,30) | Sigmoid
        b : 30      | Square
        l : 31      | Sigmoid"""

        x = torch.cat((x, s), dim=1)
        x = self.conv(x)

        f = x[:, :24] # Using linear
        a = x[:, 24:27] ** 2
        c = self.sig(x[:, 27:30])
        b = x[:, 30].unsqueeze(1) ** 2
        l = self.sig(x[:, 31].unsqueeze(1))

        x = torch.cat((f, a, c, b, l), dim=1)

        return x