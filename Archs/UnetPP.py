import torch
import torch.nn as nn
import torch.nn.functional as F

## Unet++
__all__ = ['UnetPP']

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

    
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UnetPP(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size = 256, deep_supervision=False):
        super(UnetPP, self).__init__()
    
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        nb_filters = [64, 128, 256, 512, 1024]
        

        self.b00 = DoubleConv(input_channels, nb_filters[0])
        self.b10 = DownSampling(nb_filters[0], nb_filters[1])
        self.b20 = DownSampling(nb_filters[1], nb_filters[2])
        self.b30 = DownSampling(nb_filters[2], nb_filters[3])
        self.b40 = DownSampling(nb_filters[3], nb_filters[4])


        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.b01 = DoubleConv(nb_filters[0]+ nb_filters[1],nb_filters[0])
        self.b11 = DoubleConv(nb_filters[1]+ nb_filters[2],nb_filters[1])
        self.b21 = DoubleConv(nb_filters[2]+ nb_filters[3],nb_filters[2])
        self.b31 = DoubleConv(nb_filters[3]+ nb_filters[4],nb_filters[3])

        self.b02 = DoubleConv(nb_filters[0]*2+ nb_filters[1],nb_filters[0])
        self.b12 = DoubleConv(nb_filters[1]*2+ nb_filters[2],nb_filters[1])
        self.b22 = DoubleConv(nb_filters[2]*2+ nb_filters[3],nb_filters[2])

        self.b03 = DoubleConv(nb_filters[0]*3+ nb_filters[1],nb_filters[0])
        self.b13 = DoubleConv(nb_filters[1]*3+ nb_filters[2],nb_filters[1])

        self.b04 = DoubleConv(nb_filters[0]*4+ nb_filters[1],nb_filters[0])

        self.outc = OutConv(nb_filters[0], num_classes)



    def forward(self, x):
        x00 = self.b00(x)
        x10 = self.b10(x00)
        x20 = self.b20(x10)
        x30 = self.b30(x20)
        x40 = self.b40(x30)

        x01 = self.b01(torch.cat([x00, self.up(x10)], dim=1))
        x11 = self.b11(torch.cat([x10, self.up(x20)], dim=1))
        x21 = self.b21(torch.cat([x20, self.up(x30)], dim=1))
        x31 = self.b31(torch.cat([x30, self.up(x40)], dim=1))

        x02 = self.b02(torch.cat([x00, x01, self.up(x11)], dim=1))
        x12 = self.b12(torch.cat([x10, x11, self.up(x21)], dim=1))
        x22 = self.b22(torch.cat([x20, x21, self.up(x31)], dim=1))
        x03 = self.b03(torch.cat([x00, x01, x02, self.up(x12)], dim=1))
        x13 = self.b13(torch.cat([x10, x11, x12, self.up(x22)], dim=1))
        x04 = self.b04(torch.cat([x00, x01, x02, x03, self.up(x13)], dim=1))

        if self.deep_supervision:
            return [self.outc(x01),self.outc(x02),self.outc(x03),self.outc(x04)]
        else:
            out = self.outc(x04)
            return out
        
