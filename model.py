from model_parts import *
import torch.nn.init as init

class Autoencoder(nn.Module):
    def __init__(self, in_channel, kernel_size):
        super().__init__()

        self.subsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Encoder
        self.input = ConvReLU(in_channel, 32, kernel_size)
        self.enc1 = ConvReLU(32, 32, kernel_size)
        self.rcnn1 = RecurrentBlock(32, kernel_size)
        self.enc2 = ConvReLU(32, 43, kernel_size)
        self.rcnn2 = RecurrentBlock(43, kernel_size)
        self.enc3 = ConvReLU(43, 57, kernel_size)
        self.rcnn3 = RecurrentBlock(57, kernel_size)
        self.enc4 = ConvReLU(57, 76, kernel_size)
        self.rcnn4 = RecurrentBlock(76, kernel_size)
        self.enc5 = ConvReLU(76, 101, kernel_size)
        self.rcnn5 = RecurrentBlock(101, kernel_size)
        self.enc6 = ConvReLU(101, 101, kernel_size)
        self.rcnn6 = RecurrentBlock(101, kernel_size)

        # Decoder
        self.dec1 = DecoderBlock(202, 76, 76, kernel_size)
        self.dec2 = DecoderBlock(152, 57, 57, kernel_size)
        self.dec3 = DecoderBlock(114, 43, 43, kernel_size)
        self.dec4 = DecoderBlock(86, 32, 32, kernel_size)
        self.dec5 = DecoderBlock(64, 128, 64, kernel_size)

        # Output
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=kernel_size, padding='same')

    def forward(self, x): # 128 128 7
        # Encoder
        x = self.input(x) # 128 128 32

        x = self.enc1(x)
        x, x1 = self.rcnn1(x)
        x = self.subsample(x) # 64 64 32

        x = self.enc2(x)
        x, x2 = self.rcnn2(x)
        x = self.subsample(x) # 32 32 43

        x = self.enc3(x)
        x, x3 = self.rcnn3(x)
        x = self.subsample(x) # 16 16 57

        x = self.enc4(x)
        x, x4 = self.rcnn4(x)
        x = self.subsample(x) # 8 8 76

        x = self.enc5(x)
        x, x5 = self.rcnn5(x)
        x = self.subsample(x) # 4 4 101

        x = self.enc6(x)
        x, _ = self.rcnn6(x) # 4 4 101

        # Decoder
        x = self.upsample(x)
        x = self.dec1(x, x5) # 8 8 76

        x = self.upsample(x)
        x = self.dec2(x, x4) # 16 16 57

        x = self.upsample(x)
        x = self.dec3(x, x3) # 32 32 43

        x = self.upsample(x)
        x = self.dec4(x, x2) # 64 64 32

        x = self.upsample(x)
        x = self.dec5(x, x1) # 128 128 64

        # Output
        x = self.output(x) # 128 128 3

        return x
    
    def clear_hidden_state(self):
        self.rcnn1.h = None
        self.rcnn2.h = None
        self.rcnn3.h = None
        self.rcnn4.h = None
        self.rcnn5.h = None
        self.rcnn6.h = None
