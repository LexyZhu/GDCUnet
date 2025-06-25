import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Unet']

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
    
class DownSampling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownSampling, self).__init__()
        self.down_sampling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.down_sampling(x)
    
class UpSampling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampling, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        cat_x = torch.cat([x1,x2], dim = 1)
        return self.conv(cat_x)
    
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size = 256, deep_supervision=False):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.inc = DoubleConv(input_channels, img_size)
        self.down1 = DownSampling(img_size, img_size*2)
        self.down2 = DownSampling(img_size*2, img_size*4)
        self.down3 = DownSampling(img_size*4, img_size*8)

        self.down4 = DownSampling(img_size*8, img_size*16)

        self.up1 = UpSampling(img_size*16,img_size*8)
        self.up2 = UpSampling(img_size*8,img_size*4)
        self.up3 = UpSampling(img_size*4,img_size*2)
        self.up4 = UpSampling(img_size*2,img_size)
        self.out = OutConv(img_size, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits
        
        

