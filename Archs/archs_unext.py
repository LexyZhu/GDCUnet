import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
__all__ = ['Unext']


class encoder(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.encode(x)
    

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, stride=4, in_chans=3, embed_dim=768):
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
        x = self.norm(x)

        return x, H, W
    
    
class eTok_MLP(nn.Module):
    def __init__(self, img_size, in_chan, embed_dim, block):
        super(eTok_MLP, self).__init__()
        self.patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=2, in_chans=in_chan,
                                             embed_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.block = block
        
    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        for i, blk in enumerate(self.block):
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class Decoder(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Decoder, self).__init__()
        self.convb = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_chan)
        )
    def forward(self, x1, x2):
        x1 = self.convb(x1)
        x1 = F.interpolate(x1, scale_factor=(2,2), mode='bilinear')
        x1 = F.relu(x1, inplace=True)
        return torch.add(x1, x2)

class dTok_MLP(nn.Module):
    def __init__(self, in_chan, out_chan, block):
        super(dTok_MLP, self).__init__()
        self.decode = Decoder(in_chan, out_chan)
        self.block = block
        self.norm = nn.LayerNorm(out_chan)
        
    def forward(self, x1, x2):
        B = x1.shape[0]
        x1 = self.decode(x1, x2)
        _,_,H,W = x1.shape
        x1 = x1.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.block):
            x1 = blk(x1, H, W)
        
        x1 = self.norm(x1)
        return x1.reshape(B,H,W,-1).permute(0, 3, 1, 2).contiguous()

class final(nn.Module):
    def __init__(self, in_chan, out_chan, num_classes):
        super(final, self).__init__()
        self.convb = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_chan)
        )
        self.conve = nn.Conv2d(out_chan, num_classes, kernel_size=1)
    def forward(self, x):
        x = self.convb(x)
        x = F.interpolate(x, scale_factor=(2,2), mode='bilinear')
        x = F.relu(x, inplace=True)
        return self.conve(x)
    
class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2
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

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x
    
class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
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

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
    

        

class Unext(nn.Module):
    def __init__(self, num_classes = 1, in_channel = 3, img_size = 256, embed_dims = [ 128, 160, 256], deep_supervision=False, patch_size=16,
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super(Unext, self).__init__()
        self.encode1 = encoder(in_channel, 16)
        self.encode2 = encoder(16, 32)
        self.encode3 = encoder(32, 128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[2], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=nn.LayerNorm,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=nn.LayerNorm,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[2], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=nn.LayerNorm,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[2], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=nn.LayerNorm,
            sr_ratio=sr_ratios[0])])
        
        self.etok_mlp = eTok_MLP(img_size=img_size // 4, in_chan = embed_dims[0], 
                                embed_dim = embed_dims[1], block = self.block1)
        self.btok_mlp = eTok_MLP(img_size=img_size // 8, in_chan = embed_dims[1], 
                                 embed_dim = embed_dims[2], block = self.block2)
        self.btok_mlp2 = dTok_MLP(in_chan = embed_dims[2], out_chan = embed_dims[1], block = self.dblock1)
        self.dtok_mlp = dTok_MLP(in_chan = embed_dims[1], out_chan = embed_dims[0], block = self.dblock2)

        self.decode3 = Decoder(128,32)
        self.decode2 = Decoder(32, 16)
        self.decode1 = Decoder(16,16)
        self.final = final(in_chan=16, out_chan=16,num_classes=1)


    def forward(self, x):
        t1 = self.encode1(x)
        t2 = self.encode2(t1)
        t3 = self.encode3(t2)
        t4 = self.etok_mlp(t3)
        out = self.btok_mlp(t4)

        out = self.btok_mlp2(out,t4)
        out = self.dtok_mlp(out, t3)
        out = self.decode3(out,t2)
        out = self.decode2(out,t1)
        out = self.final(out)
        return out
        
