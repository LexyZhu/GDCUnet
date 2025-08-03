import torch
import torch.nn.functional as F
from utils import *

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm import create_model
import math
import numpy
from einops import rearrange
import torchvision.models as models
from torch.nn.modules.utils import _pair
__all__ = ['GDCUnet']


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):      
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):

        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 每个 [b, n, d]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
     
        return out

class Offset_net(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, group=1, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.embedding_in = nn.Linear(dim, dim*2, bias=False)
        self.embedding_out = nn.Linear(dim*2, dim, bias=False)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim*2, Attention(dim*2, heads=heads, dropout=dropout))),

                Residual(PreNorm(dim*2, FeedForward(dim*2, mlp_dim, dropout=dropout)))
            ]))
        self.group = group

    def forward(self, x, mask=None):
        bs_gp, dim, wid, hei = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        bs = bs_gp // self.group
        gp = self.group
        x = x.reshape(bs, gp, dim, wid, hei)
        x = x.permute(0, 1, 3, 4, 2).reshape(bs, gp * wid * hei, dim)
        x = self.embedding_in(x)
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        x = self.embedding_out(x)
        x = x.reshape(bs, gp, wid, hei, dim).permute(0, 1, 4, 2, 3).reshape(bs_gp, dim, wid, hei)
        return x


class SAFDConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SAFDConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        # Offset transformer layer
        self.offset_transformer = Offset_net(in_channels, 1, 4, 64)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        for p in self.offset_transformer.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.constant_(p, 0)

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        # Generate learned offsets
        offset = self.offset_transformer(x)
        b, _, h, w = offset.size()
        N, C, H, W = x.size()

        # Create sampling grid
        grid = self._make_grid(N, H, W, dtype, device)

        # Calculate offset grid for deformable convolution
        offset = self._get_offset_grid(offset, grid, device)

        x = F.grid_sample(x, offset, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Perform the convolution
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return x

    def _make_grid(self, N, H, W, dtype, device):
        h_range = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype)
        w_range = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(h_range, w_range)
        grid = torch.stack((grid_w, grid_h), dim=-1)
        grid = grid.expand(N, -1, -1, -1)
        return grid

    def _get_offset_grid(self, offset, grid, device):
        N, _, H, W = offset.size()
        offset = offset.permute(0, 2, 3, 1).contiguous()
        offset = offset.view(N, H, W, -1, 2)
        offset = offset.mean(dim=3)  # Reduce the offset if it has multiple channels per location
        final_grid = grid + offset
        return final_grid


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, H, W):
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x


class Feature_Incentive_Block(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)
        return x, H, W


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class SAFD_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SAFD_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            SAFDConv(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SAFDConv(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
    
class SAFD_D_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SAFD_D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            SAFDConv(in_ch, in_ch, 5, padding=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            SAFDConv(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class  GDCUnet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
                 embed_dims=[16, 32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.conv1 = DoubleConv(input_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = SAFD_DoubleConv(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)

        self.pool4 = nn.MaxPool2d(2)

        self.FIBlock1 = Feature_Incentive_Block(img_size=img_size // 4, patch_size=3, stride=1,
                                                in_chans=embed_dims[2],
                                                embed_dim=embed_dims[3])
        self.FIBlock2 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[3],
                                                embed_dim=embed_dims[4])
        self.FIBlock3 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[4],
                                                embed_dim=embed_dims[3])

        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])
        self.norm3 = norm_layer(embed_dims[3])

        self.FIBlock4 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])

        self.decoder3 = SAFD_D_DoubleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_DoubleConv(embed_dims[1], embed_dims[0])
        self.decoder1 = D_DoubleConv(embed_dims[0], 8)

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        ### Conv Stage
        out = self.conv1(x)
        t1 = out

        out = self.pool1(out)
        out = self.conv2(out)

        t2 = out
        out = self.pool2(out)
        out = self.conv3(out)

        t3 = out
        out = self.pool3(out)

        ### Stage 4
        out, H, W = self.FIBlock1(out)

        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        out = self.pool4(out)

        ### Bottleneck
        out, H, W = self.FIBlock2(out)

        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out, H, W = self.FIBlock3(out)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')

        ### Stage 4
        out = torch.add(out, t4)
        out = out.flatten(2).transpose(1, 2)

        out = self.norm3(out)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(F.relu(self.dbn4(self.FIBlock4(out))), scale_factor=(2, 2), mode='bilinear')

        ### Conv Stage
        out = torch.add(out, t3)
        out = F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t2)
        out = F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t1)
        out = self.decoder1(out)

        out = self.final(out)

        return out
