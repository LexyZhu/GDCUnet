import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResUnet']

class ResBlock(nn.Module):
    def __init__(self, in_chan, out_chan, dilation_rates):
        super(ResBlock, self).__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.BatchNorm2d(in_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_chan, out_channels=out_chan, 
                          kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_chan, out_channels=out_chan, 
                          kernel_size=3, padding=d, dilation=d)
            ) for d in dilation_rates
        )
    def forward(self, x):
        out = x
        for branch in self.branches:
            out = out + branch(x)
        return out
    
class Conv2DN(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride, padding):
        super(Conv2DN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                              kernel_size=kernel_size, stride=stride, padding = padding)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return self.bn(self.conv(x))
            

class PSP_Pooling(nn.Module):
    def __init__(self, in_chan, out_chan, sizes = [1,2,4,8]):
        super(PSP_Pooling, self).__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.MaxPool2d(kernel_size=s,stride=s),
                nn.Upsample(scale_factor=s, mode='nearest'),
                Conv2DN(in_chan, out_chan // 4, kernel_size=1, stride=1, padding=0)
            ) for s in sizes
        ) 
        self.conv2dn = Conv2DN(out_chan, out_chan, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        out = []
        # print(x.shape)
        for branch in self.branches:
            m = branch(x)
            out.append(m)
            # print(m.shape)
            # print(m.shape)
        out = torch.cat(out, dim=1)
        # print(out.shape)
        # print(out.shape)
        out = self.conv2dn(out)
        return out
    
class Combine(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Combine, self).__init__()
        self.up = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv2n = Conv2DN(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
    def forward(self, r, x):
        # print(x.shape, r.shape)
        x = self.up(x)
        x = F.relu(x,inplace=True)
        # print(x.shape)
        x = torch.cat([x,r], dim=1)
        x = self.conv2n(x)
        return x
        
class ResUnet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, img_size=256, deep_supervision=False):
        super(ResUnet, self).__init__()
        self.n_filters = [32, 64, 128, 256, 512, 1024]
        self.conv2dn1 = nn.Sequential(
            nn.Conv2d(3,self.n_filters[0], kernel_size=3,padding=1),
            nn.BatchNorm2d(self.n_filters[0])
        )
        self.relu = nn.ReLU(inplace=True)
        self.resb1 = ResBlock(self.n_filters[0], self.n_filters[0], dilation_rates=[1, 3, 15, 31])
        self.conv1 = nn.Conv2d(self.n_filters[0], self.n_filters[1], kernel_size=3, stride=2, padding=1)
        self.resb2 = ResBlock(self.n_filters[1], self.n_filters[1], dilation_rates=[1, 3, 15, 31])
        self.conv2 = nn.Conv2d(self.n_filters[1], self.n_filters[2], kernel_size=3, stride=2, padding=1)
        self.resb3 = ResBlock(self.n_filters[2], self.n_filters[2], dilation_rates=[1, 3, 15])
        self.conv3 = nn.Conv2d(self.n_filters[2], self.n_filters[3], kernel_size=3, stride=2, padding=1)
        self.resb4 = ResBlock(self.n_filters[3], self.n_filters[3], dilation_rates=[1, 3, 15])

        self.conv4 = nn.Conv2d(self.n_filters[3], self.n_filters[4], kernel_size=3, stride=2, padding=1)
        self.resb5 = ResBlock(self.n_filters[4], self.n_filters[4], dilation_rates=[1])

        # self.conv5 = nn.Conv2d(self.n_filters[4], self.n_filters[5], kernel_size=3, stride=2, padding=1)
        # self.resb6 = ResBlock(self.n_filters[5], self.n_filters[5], dilation_rates=[1])

        self.pspb = PSP_Pooling(self.n_filters[4], self.n_filters[4])

        # self.comb5 = Combine(self.n_filters[5] + self.n_filters[4], self.n_filters[4])
        self.comb4 = Combine(self.n_filters[4] + self.n_filters[3], self.n_filters[3])
        self.comb3 = Combine(self.n_filters[3] + self.n_filters[2], self.n_filters[2])
        self.comb2 = Combine(self.n_filters[2] + self.n_filters[1], self.n_filters[1])
        self.comb1 = Combine(self.n_filters[1] + self.n_filters[0], self.n_filters[0])
        self.conv0 = Conv2DN(in_chan = self.n_filters[0]*2, out_chan = self.n_filters[0], kernel_size=1, stride=1, padding=0)
        self.pspl = PSP_Pooling(self.n_filters[0]*2, self.n_filters[0])
        self.distance_logits = nn.Sequential(
            Conv2DN(self.n_filters[1], self.n_filters[0], kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            Conv2DN(self.n_filters[0], self.n_filters[0], kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_filters[0], 1, kernel_size=1, padding=0)
        )
        self.bound_logits = nn.Sequential(
            Conv2DN(self.n_filters[0]+1, self.n_filters[0], kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_filters[0], 1, kernel_size=1, padding=0)
        )

        self.logits = nn.Sequential(
            Conv2DN(self.n_filters[0]+2, self.n_filters[0], kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            Conv2DN(self.n_filters[0], self.n_filters[0], kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_filters[0], 1, kernel_size=1, padding=0)
        )
        # self.color_logits = nn.Conv2d(self.n_filters[0], 3, kernel_size=1, padding=0)

        self.out_conv = nn.Conv2d(self.n_filters[0], num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv2dn1(x)
        r0 = self.relu(x)
        r1 = self.resb1(r0)

        x = self.conv1(r1)
        r2 = self.resb2(x)

        x = self.conv2(r2)
        r3 = self.resb3(x)

        x = self.conv3(r3)
        r4 = self.resb4(x)

        x = self.conv4(r4)
        r5 = self.resb5(x)

        x = self.pspb(r5)

        x = self.comb4(r4, x)
        x = self.resb4(x)
        x = self.comb3(r3, x)
        x = self.resb3(x)
        x = self.comb2(r2, x)
        x = self.resb2(x)
        x = self.comb1(r1, x)
        x = self.resb1(x)
        x1 = torch.cat([x, r0], dim=1)


        x = self.pspl(x1)
        x = self.out_conv(x)
        return x
