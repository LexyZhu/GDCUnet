import torch
import torch.nn as nn
import torch.nn.functional as F

## Unet
__all__ = ['Attention_Unet']

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
    

    
class AttGate(nn.Module):
    def __init__(self, in_chan, gating_chan):
        super(AttGate, self).__init__()
        
        self.wg = nn.Conv2d(in_channels=gating_chan, out_channels=in_chan//2,
                            kernel_size=(1,1), stride=(1,1), bias=False)
        self.wx = nn.Conv2d(in_channels=in_chan,out_channels=in_chan//2,
                            kernel_size=(1,1), stride=(1,1), bias=False)
        self.psi = nn.Conv2d(in_channels=in_chan//2, out_channels=1,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.final=nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=in_chan, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_chan),
        )
        # self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        for p in self.offset_transformer.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.constant_(p, 0)

    def forward(self, x, g):
        input_size = x.shape
        batch_size = input_size[0]
        assert batch_size == g.shape[0]

        Wx = self.wx(x)
        Wx_size = Wx.shape
        Wg = self.wg(g)
        Wg = F.upsample(Wg, size=Wx_size[2:], mode='bilinear')
        f = F.relu(Wg+Wx,inplace=True)

        f = self.psi(f)
        f = F.sigmoid(f)
        f = F.upsample(f, size=input_size[2:], mode='bilinear')
        f = f.expand_as(x)*x
        output = self.final(f)

        return output, f


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

class Attention_Unet(nn.Module):
    def __init__(self, num_classes, input_channels=3, img_size = 256, deep_supervision=False):
        super(Attention_Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.inc = DoubleConv(input_channels, img_size)
        self.down1 = DownSampling(img_size, img_size*2)
        self.down2 = DownSampling(img_size*2, img_size*4)
        self.down3 = DownSampling(img_size*4, img_size*8)

        self.ag3 = AttGate(img_size*4, img_size*8)
        self.up3 = UpSampling(img_size*8,img_size*4)
        self.ag2 = AttGate(img_size*2, img_size*4)
        self.up2 = UpSampling(img_size*4,img_size*2)
        self.ag1 = AttGate(img_size, img_size*2)
        self.up1 = UpSampling(img_size*2,img_size)
        self.out = OutConv(img_size, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)

        x3, att1 = self.ag3(x3, x)
        x = self.up3(x, x3)
        x2, att2 = self.ag2(x2, x)
        x = self.up2(x, x2)
        x1, att3 = self.ag1(x1, x)
        x = self.up1(x, x1)
        x = self.out(x)
        return x

        