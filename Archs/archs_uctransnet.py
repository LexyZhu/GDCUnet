import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import copy

## Unet
__all__ = ['UctransUnet']

class ConvBatchNormReLU(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(ConvBatchNormReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_chan)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            ConvBatchNormReLU(in_ch, out_ch),
            ConvBatchNormReLU(out_ch, out_ch)
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

class CCT_embedding(nn.Module):
    def __init__(self, patch_size, in_chan, img_size):
        super(CCT_embedding, self).__init__()
        n_patches = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(in_chan, in_chan, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_chan))
        self.dropout = nn.Dropout(0.1) # Need to check this value

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Reconstruct(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size==3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_chan)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
    def forward(self, x):
        B, n_patch, hidden = x.shape
        h, w = int(n_patch ** 0.5), int(n_patch ** 0.5)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Attention(nn.Module):
    def __init__(self, channel_num, kv_size=960, num_attention_heads=4):
        super(Attention, self).__init__()
        self.kv_size = kv_size # check this value
        self.channel_nums = channel_num
        self.num_attention_heads = num_attention_heads

        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(num_attention_heads):
            query = nn.Linear(channel_num, channel_num, bias = False)
            key = nn.Linear(self.kv_size, self.kv_size, bias = False)
            value = nn.Linear(self.kv_size, self.kv_size, bias = False)
            self.query.append(copy.deepcopy(query))
            self.key.append(copy.deepcopy(key)) 
            self.value.append(copy.deepcopy(value)) 

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = nn.Softmax(dim=3)
        self.out = nn.Linear(channel_num, channel_num, bias = False)
        self.attn_dropout = nn.Dropout(0.1)  # Need to check this value
        self.proj_dropout = nn.Dropout(0.1)  # Need to check this value

    def forward(self, emb, emb_all):
        multi_head_query = []
        multi_head_key = []
        multi_head_value = []
        if emb is not None:
            for query in self.query:
                Q = query(emb)
                multi_head_query.append(Q)
        for key in self.key:
            K = key(emb_all)
            multi_head_key.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_value.append(V)
        multi_head_Q = torch.stack(multi_head_query, dim=1).transpose(-1, -2)
        multi_head_K = torch.stack(multi_head_key, dim=1)
        multi_head_V = torch.stack(multi_head_value, dim=1)
        attention_scores = torch.matmul(multi_head_Q, multi_head_K) / (self.kv_size ** 0.5)
        attention_probs = self.attn_dropout(self.softmax(self.psi(attention_scores)))
        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer = torch.matmul(attention_probs, multi_head_V)
        context_layer = context_layer.permute(0, 3, 2, 1).contiguous()
        context_layer = context_layer.mean(dim = 3)

        out = self.out(context_layer)
        out = self.proj_dropout(out)
        return out

class mlp(nn.Module):
    def __init__(self, in_chan, mlp_chan):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(in_chan, mlp_chan)
        self.fc2 = nn.Linear(mlp_chan, in_chan)
        self.act_fc = nn.GELU()
        self.dropout = nn.Dropout(0.1)  # Need to check this value

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class CCT(nn.Module):
    def __init__(self, channel_num, expand_ratio=4):
        super(CCT, self).__init__()
        self.channel_num = channel_num
        self.expand_ratio = expand_ratio
        self.norm = nn.LayerNorm(channel_num)
        self.norm_all = nn.LayerNorm(960)
        self.channel_attn = Attention(channel_num)
        self.mlp = mlp(channel_num, channel_num * expand_ratio)
    def forward(self, embed, embed_all):
        out1 = self.norm(embed)
        out_all = self.norm_all(embed_all)
        out2 = self.channel_attn(out1, out_all)
        out2 = out2 + out1

        out3 = self.norm(out2)        
        out3 = self.mlp(out3)
        out3 = out3 + out2
        return out3
    

class CCTs(nn.Module):
    def __init__(self, channel_num, num_layers=4):
        super(CCTs, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = CCT(channel_num)
            self.layers.append(copy.deepcopy(layer))
        self.norm = nn.LayerNorm(channel_num, eps=1e-6)

    def forward(self, embed, embed_all):
        for layer in self.layers:
            embed = layer(embed, embed_all)
        embed = self.norm(embed)
        return embed



class Full_connection(nn.Module):
    def __init__(self, channel_nums=[64, 128, 256, 512], patch_size=[16, 8, 4, 2], num_layers=4,img_size=256):
        super(Full_connection, self).__init__()
        self.patchsize1 = patch_size[0]
        self.patchsize2 = patch_size[1]
        self.patchsize3 = patch_size[2]
        self.patchsize4 = patch_size[3]

        self.embed1 = CCT_embedding(patch_size=self.patchsize1, in_chan=channel_nums[0],img_size=img_size)
        self.embed2 = CCT_embedding(patch_size=self.patchsize2, in_chan=channel_nums[1],img_size=img_size//2)
        self.embed3 = CCT_embedding(patch_size=self.patchsize3, in_chan=channel_nums[2],img_size=img_size//4)
        self.embed4 = CCT_embedding(patch_size=self.patchsize4, in_chan=channel_nums[3],img_size=img_size//8)
        
        # ViT layers
        self.vit1 = CCTs(channel_nums[0])
        self.vit2 = CCTs(channel_nums[1])
        self.vit3 = CCTs(channel_nums[2])
        self.vit4 = CCTs(channel_nums[3])


        self.reconstruct1 = Reconstruct(channel_nums[0], channel_nums[0], kernel_size=3, scale_factor=self.patchsize1)
        self.reconstruct2 = Reconstruct(channel_nums[1], channel_nums[1], kernel_size=3, scale_factor=self.patchsize2)
        self.reconstruct3 = Reconstruct(channel_nums[2], channel_nums[2], kernel_size=3, scale_factor=self.patchsize3)
        self.reconstruct4 = Reconstruct(channel_nums[3], channel_nums[3], kernel_size=3, scale_factor=self.patchsize4)

    def forward(self, e1, e2, e3, e4):
        embed1 = self.embed1(e1)
        embed2 = self.embed2(e2)
        embed3 = self.embed3(e3)
        embed4 = self.embed4(e4)
        embed_all = torch.cat([embed1, embed2, embed3, embed4], dim=2)

        encoded1 = self.vit1(embed1, embed_all)
        encoded2 = self.vit2(embed2, embed_all)
        encoded3 = self.vit3(embed3, embed_all)
        encoded4 = self.vit4(embed4, embed_all)

        out1 = self.reconstruct1(encoded1)
        out2 = self.reconstruct2(encoded2)
        out3 = self.reconstruct3(encoded3)
        out4 = self.reconstruct4(encoded4)

        # Should be concatenated, but here they just add them (????????)
        out1 = out1+e1
        out2 = out2+e2
        out3 = out3+e3
        out4 = out4+e4

        return out1, out2, out3, out4
    

class CCA(nn.Module):
    def __init__(self, F_o, F_d):
        super(CCA, self).__init__()
        self.go = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F_o, F_d),
        )
        self.gd = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F_d, F_d),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, o, d):
        avg_pool_o = F.avg_pool2d(o, kernel_size = o.shape[2:], stride = o.shape[2:])
        channel_att_o = self.go(avg_pool_o)
        avg_pool_d = F.avg_pool2d(d, kernel_size = d.shape[2:], stride = d.shape[2:])
        channel_att_d = self.gd(avg_pool_d)
        channel_att_sum = (channel_att_o + channel_att_d)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(d)
        o_hat = d * scale
        o_hat = self.relu(o_hat)
        return o_hat

    
class UpSampling_CCA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampling_CCA, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.cca = CCA(F_o = in_ch//2, F_d = in_ch//2)
        self.conv1 = ConvBatchNormReLU(in_ch, out_ch)
        self.conv2 = ConvBatchNormReLU(out_ch, out_ch)
    
    def forward(self, o, d):
        o = self.up(o)
        o_hat = self.cca(o, d)
        cat_d = torch.cat([d, o_hat],dim = 1)
        out = self.conv1(cat_d)
        out = self.conv2(out)
        return out
    
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UctransUnet(nn.Module):
    def __init__(self, num_classes, base_channel = 64, input_channels=3, img_size = 256, deep_supervision=False):
        super(UctransUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.inc = ConvBatchNormReLU(input_channels, base_channel)
        self.down1 = DownSampling(base_channel, base_channel*2)
        self.down2 = DownSampling(base_channel*2, base_channel*4)
        self.down3 = DownSampling(base_channel*4, base_channel*8)
        self.down4 = DownSampling(base_channel*8, base_channel*8)

        self.mtc = Full_connection(channel_nums=[base_channel, base_channel*2, base_channel*4, base_channel*8],
                                patch_size=[16, 8, 4, 2], num_layers=4)
        

        self.up4 = UpSampling_CCA(base_channel*16,base_channel*4)
        self.up3 = UpSampling_CCA(base_channel*8,base_channel*2)
        self.up2 = UpSampling_CCA(base_channel*4,base_channel)
        self.up1 = UpSampling_CCA(base_channel*2,base_channel)
        self.outc = OutConv(base_channel, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1, x2, x3, x4 = self.mtc(x1, x2, x3, x4)
        # print(x5.shape, x4.shape)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits
        