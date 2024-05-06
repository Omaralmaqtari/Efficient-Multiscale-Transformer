import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from loss import *

from math import pi
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
import sys
import time

# gaussian kernels
def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue 
    g = gaussian_filter(n,sigma)
    gaussiankernel = torch.from_numpy(g)
    
    return gaussiankernel.float()

# laplaceian kernels
def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue
    l = laplace(n)
    laplacekernel = torch.from_numpy(l)
    
    return laplacekernel.float()


# Squeeze and Excitation Attention
class SEM(nn.Module):
    def __init__(self, ch_out, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out//reduction, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch_out//reduction, ch_out, kernel_size=1,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        
        return x * y.expand_as(x)


# Convolution layer
class iConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernal, padding, groups, norm_groups):
        super().__init__()
        self.iconv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernal, padding=padding, groups=groups, bias=False),
            nn.PReLU(num_parameters=ch_out, init=-0.01),
            nn.GroupNorm(norm_groups,ch_out)
            )
            
    def forward(self, x):
        return self.iconv(x)
    

# Fourier Feature Mapping 
class FFM(nn.Module):
    def __init__(self, input_channels, mapping_size, scale=20):
        super(FFM, self).__init__()

        self.mapping_size = mapping_size
        self._B = torch.randn((input_channels, mapping_size)) * scale

    def forward(self, x):
        batches, channels, width, height = x.shape
        
        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self.mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


# Edge Extraction Module
class EEM(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, reduction):
        super(EEM, self).__init__()
        
        self.gk = gaussiankernel(ch_in, ch_in, kernel, kernel-2, 0.9)
        self.lk = laplaceiankernel(ch_in, ch_in, kernel, 0.9)
        self.gk = nn.Parameter(self.gk, requires_grad=False)
        self.lk = nn.Parameter(self.lk, requires_grad=False)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.01),
            nn.InstanceNorm2d(int(ch_out/2))
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.01),
            nn.InstanceNorm2d(int(ch_out/2))
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(ch_out/2), ch_out, kernel_size=3,padding=1,groups=4,bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.02),
            nn.GroupNorm(4,ch_out),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_out, ch_out, kernel_size=1,padding=0,groups=4,bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.02),
            )
        
        self.sem1 = SEM(ch_out,reduction=reduction)
        self.sem2 = SEM(ch_out,reduction=reduction)
        self.prelu = nn.PReLU(num_parameters=ch_in, init=0.01)
        
    def forward(self, x):
        DoG = F.conv2d(x, self.gk.to('cuda:0'), padding=1,groups=1)
        LoG = F.conv2d(DoG, self.lk.to('cuda:0'), padding=1,groups=1)
        DoG = self.conv1(DoG-x)
        LoG = self.conv2(LoG)
        tot = self.conv3(DoG+LoG)
        tot1 = self.sem1(tot)
        
        x1 = self.sem2(x)
        
        return self.prelu(x+x1+tot+tot1)
    
# Parallel Feature Module
class PFM(nn.Module):
    def __init__(self, ch_in, ch_out, reduction):
        super(PFM, self).__init__()
        # reducer
        self.reducer = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.02),
            nn.GroupNorm(4,ch_out)
            )
        
        # 3x3 conv branch
        self.c1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,padding=1,dilation=1,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4,ch_out)
            )
        self.c2 = nn.Sequential(
            nn.Conv2d(ch_out, int(ch_out*3), kernel_size=3,padding=1,dilation=1,groups=2,bias=False),
            nn.PReLU(num_parameters=int(ch_out*3), init=0.02),
            nn.GroupNorm(2,int(ch_out*3))
            )
        # 3x3 pool branch
        self.c3 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=1,padding=0,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4,ch_out),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_out, ch_out, kernel_size=1,padding=0,groups=2,bias=False),
            nn.ReLU(True)
            )

        self.EEM = EEM(ch_out,ch_out,kernel=3,reduction=reduction[0])
        self.sem1 = SEM(ch_in,reduction=reduction[1])
        self.sem2 = SEM(ch_in,reduction=reduction[1])
        self.prelu = nn.PReLU(num_parameters=ch_in, init=0.01)

    def forward(self, x):
        x1 = self.reducer(x)
        x2 = self.c1(x1) + self.c3(x1) + self.EEM(x1)

        y1 = torch.cat([self.c2(x1+x2),x1+x2],1)
        y2 = self.sem1(y1)
        x1 = self.sem2(x)
        
        return self.prelu(x+x1+y1+y2)

    
# Local Self Attention
class LSA(nn.Module):
    def __init__(self, ch_in, head, ps, dropout):
        super(LSA, self).__init__()
        self.c = ch_in
        self.H = head
        self.ph = int(ps)
        self.pw = int(ps)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in*head*2, kernel_size=3, padding=1,groups=4,bias=True),#3
            nn.ReLU(True),
            nn.GroupNorm(2,ch_in*head*2)#3
            )

        self.to_out = nn.Sequential(
            nn.Conv2d((ch_in*head), ch_in, kernel_size=1, padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_in, init=-0.01),
            nn.GroupNorm(1,ch_in),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_in, ch_in, kernel_size=1, padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_in, init=-0.01),
            )

    def forward(self, x):
        _, _, h, _ = x.shape
        
        qkv = rearrange(self.to_qkv(x), 'b (c H) h w -> b H c h w', H=self.H)
        qkv = rearrange(qkv, 'b H (c a) h w -> b H c a h w', c=self.c).contiguous().chunk(2, dim = 3)#3
    
        q, k = map(lambda t: rearrange(t, 'b H c a (ph h) (pw w) -> b H (ph pw) (c h w a)', ph=self.ph, pw=self.pw).contiguous(), qkv)#v
        v = (q + k)/2
        
        _, _, p, _ = k.shape
        q = torch.matmul(q, k.transpose(-1, -2)) * (p ** -0.5)
        
        q = self.dropout(self.softmax(q))
        qkv = torch.matmul(q, v)
        
        qkv = rearrange(qkv, 'b H p (c d) -> b H c p d', c=self.c)
        qkv = rearrange(qkv, 'b H c (ph pw) (h w) -> b (c H) (ph h) (pw w)', ph=self.ph, h=int(h/self.ph)).contiguous()
        
        return self.to_out(qkv)
    
# Global Self Attention    
class GSA(nn.Module):
    def __init__(self, ch_in, head, ps, dropout):
        super(GSA, self).__init__()
        self.c = ch_in
        self.H = head
        self.ph = ps
        self.pw = ps
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in*head*2, kernel_size=3, padding=1,groups=4,bias=True),#3
            nn.ReLU(True),
            nn.GroupNorm(2,ch_in*head*2)#3
            )
        
        self.to_out = nn.Sequential(
            nn.Conv2d((ch_in*head), ch_in, kernel_size=1,padding=0,groups=4,bias=False),
            nn.PReLU(num_parameters=ch_in, init=0.01),
            nn.GroupNorm(1,ch_in),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_in, ch_in, kernel_size=1,padding=0,groups=4,bias=False),
            nn.PReLU(num_parameters=ch_in, init=0.01)
            )

    def forward(self, x):
        _, _, h, _ = x.shape
        
        qkv = rearrange(self.to_qkv(x), 'b (c H) h w -> b H c h w', H=self.H)
        qkv = rearrange(qkv, 'b H (c a) h w -> b H c a h w', c=self.c).contiguous().chunk(2, dim = 3)#3

        q, k = map(lambda t: rearrange(t, 'b H c a (ph h) (pw w) -> b H (ph pw) (c h w a)', ph=self.ph, pw=self.pw).contiguous(), qkv)#v
        v = (q + k)/2
        
        _, _, p, _ = k.shape
        q = torch.matmul(q, k.transpose(-1, -2)) * (p ** -0.5)
        
        q = self.dropout(self.softmax(q))
        qkv = torch.matmul(q, v)
        
        qkv = rearrange(qkv, 'b H p (c d) -> b H c p d', c=self.c)
        qkv = rearrange(qkv, 'b H c (ph pw) (h w) -> b (c H) (ph h) (pw w)', ph=self.ph, h=int(h/self.ph)).contiguous()
        
        return self.to_out(qkv)
    

class Transformerblock(nn.Module):
    def __init__(self, ch_in, mid_ch, head, patch_size, r, feat, reduction, dropout, depth):
        super(Transformerblock, self).__init__()
        self.ps = patch_size
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                iConv(ch_in, mid_ch[i], 3, 1, 8, 4),
                PFM(mid_ch[i], int(mid_ch[i]/4), reduction),
                iConv(mid_ch[i], feat, 3, 1, 4, 4),
                LSA(feat, head, self.ps*r, dropout),#
                GSA(feat, head, self.ps, dropout),
                iConv(feat, mid_ch[i], 1, 0, 2, 2)
                ]))
            
            ch_in = mid_ch[i]
        
    def forward(self, x):
        out = []
        
        for iconv, PFM, iconv1, LSA, GSA, iconv2 in self.layers: #LSA,
            x = iconv(x)
            x =  PFM(x) + x
            x = iconv2(LSA(iconv1(x)) * GSA(iconv1(x))) + x #LSA(iconv1(x)) *
            out.append(x)
            
        return out
 

class EMT(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(EMT, self).__init__()
        
        self.prelayer = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.sem = SEM(64, reduction=16)
        
        self.T1 = Transformerblock(64, ch_in[0], 4, 8, 2, 12, [6,24], 0.05, depth=2)
        self.T2 = Transformerblock(96, ch_in[1], 6, 4, 2, 20, [8,32], 0.025, depth=3)
        self.T3 = Transformerblock(128, ch_in[2], 8, 2, 2, 28, [12,48], 0.0125, depth=4)
        self.T4 = nn.ModuleList([])
        self.T4.append(nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(192, ch_in[3][0], kernel_size=3, padding=1, dilation=1, groups=4, bias=False),
                nn.PReLU(num_parameters=ch_in[3][0], init=0.),
                nn.GroupNorm(2,ch_in[3][0])),
            nn.Sequential(
                nn.Conv2d(192, ch_in[3][1], kernel_size=3, padding=2, dilation=2, groups=2, bias=False),
                nn.PReLU(num_parameters=ch_in[3][1], init=0.),
                nn.GroupNorm(2,ch_in[3][1])),
            nn.Sequential(
                nn.Conv2d(192, ch_in[3][2], kernel_size=3, padding=4, dilation=4, groups=4, bias=False),
                nn.PReLU(num_parameters=ch_in[3][2], init=0.),
                nn.GroupNorm(2,ch_in[3][2])),
            nn.AdaptiveAvgPool2d(1),
            nn.Sequential(
                nn.Conv2d((128*3)+192, ch_in[3][2], kernel_size=1, padding=0, groups=2, bias=False),
                nn.PReLU(num_parameters=ch_in[3][2], init=0.),
                nn.GroupNorm(2,ch_in[3][2]))
            ]))
        
        self.D = Decoder(ch_in, ch_out, 4, depth=[2,3,4,3])
            
        self.P = nn.MaxPool2d(2,2)
        self.d1 = nn.Dropout(0.3)
        self.d2 = nn.Dropout(0.2)
        self.d3 = nn.Dropout(0.1)
    
    def forward(self, x):
        img_shape = x.shape[2:]
        
        x0 = self.prelayer(x)
        x1 = self.sem(x0)
        x0 = x0 + x1

        x1 = self.T1(x0)
        x2 = self.T2(self.d1(self.P(x1[-1])))
        x3 = self.T3(self.d2(self.P(x2[-1])))
        
        for c1, c2, c3, ap, c4 in self.T4: 
            x1_1 = c1(self.d3(self.P(x3[-1])))
            x1_2 = c2(self.d3(self.P(x3[-1])))
            x1_3 = c3(self.d3(self.P(x3[-1])))
            x1_4 = ap(self.d3(self.P(x3[-1])))
            
            x1_4 = F.interpolate(x1_4, size=x1_3.shape[2:], mode='bilinear', align_corners=False)
            x4 = torch.cat([x1_1,x1_2,x1_3,x1_4],1)
            x4 = c4(x4)
        
        SR = self.D(x0, x1, x2, x3, x4, img_shape)

        return SR
 

class Decoder(nn.Module):
    def __init__(self, ch_in, ch_out, ps, depth):
        super(Decoder, self).__init__()
        
        self.layers0 = nn.ModuleList([])
        for i in range(1):
            self.layers0.append(nn.ModuleList([
                iConv(64, 16, 1, 0, 2, 1),
                LSA(16, 4, ps*4, 0.05),
                GSA(16, 4, ps*2, 0.05)
                ]))
            
        self.layers1 = nn.ModuleList([])
        for i in range(depth[0]):
            self.layers1.append(nn.ModuleList([
                iConv(ch_in[0][i], 16, 1, 0, 2, 1),
                LSA(16, 4, ps*4, 0.05),
                GSA(16, 4, ps*2, 0.05)
                ]))
            
        self.layers2 = nn.ModuleList([])
        for i in range(depth[1]):
            self.layers2.append(nn.ModuleList([
                iConv(ch_in[1][i], 16, 1, 0, 2, 1),
                LSA(16, 4, ps*2, 0.025),
                GSA(16, 4, ps, 0.025)
                ]))
        
        self.layers3 = nn.ModuleList([])
        for i in range(depth[2]):
            self.layers3.append(nn.ModuleList([
                iConv(ch_in[2][i], 16, 1, 0, 2, 1),
                LSA(16, 3, ps, 0.0125)
                ]))
            
        self.layers4 = nn.ModuleList([])
        for i in range(1):
            self.layers4.append(nn.ModuleList([
                iConv(ch_in[3][0], 16, 1, 0, 2, 1),
                LSA(16, 2, 2, 0.0125)
                ]))
            
        ch_in = 16 + ((depth[0]+depth[1]+depth[2])*16) + 16
        
        self.sem = SEM(int(ch_in), reduction=44)#4
        self.out = nn.Sequential(
                nn.Conv2d(int(ch_in), 64, kernel_size=3,padding=1,groups=4,bias=False),
                nn.PReLU(num_parameters=64, init=-0.02),
                nn.GroupNorm(1,64),
                nn.Conv2d(64, 64, kernel_size=1,padding=0,bias=False),
                nn.PReLU(num_parameters=64, init=-0.),
                nn.GroupNorm(1,64),
                nn.Conv2d(64, ch_out, kernel_size=1,padding=0,bias=False),
                nn.PReLU(num_parameters=ch_out, init=0.)
                )
        
    def forward(self, x0, x1, x2, x3, x4, img_shape):
        out = []
        
        for (conv1, LSA, GSA) in self.layers0:
            x = conv1(x0)
            x = LSA(x) + GSA(x) + x
            out.append(x)
        for x, (conv1, LSA, GSA) in zip(x1, self.layers1):
            x = conv1(x)
            x = LSA(x) + GSA(x) + x
            out.append(x)
        for x, (conv1, LSA, GSA) in zip(x2, self.layers2):
            x = conv1(x)
            x = LSA(x) + GSA(x) + x
            out.append(x)
        for x, (conv1, LSA) in zip(x3, self.layers3):
            x = conv1(x)
            x = LSA(x) + x
            out.append(x)
            
        for (conv1, LSA) in self.layers4:
            x = conv1(x4)
            x = LSA(x) + x
            out.append(x)
            
        out1 = []
        for i in out:
            out1.append(F.interpolate(i, size=img_shape, mode='bilinear', align_corners=False))
            
        x = torch.cat(out1,1)
        x0 = self.sem(x)

        return self.out(x0+x)
    
    
class FFM_Net(nn.Module):
    def __init__(self, mapping_size):
        super(FFM_Net, self).__init__()
        
        self.ffm = FFM(1,mapping_size=mapping_size)
        self.ffmlayer1 = nn.Sequential(
            nn.Conv2d(mapping_size*2,mapping_size*2,kernel_size=1,padding=0,bias=False),
            nn.PReLU(num_parameters=mapping_size*2, init=0.01),
            nn.GroupNorm(1,mapping_size*2)
            )

        self.ffmlayer2 = nn.Sequential(
            nn.Conv2d(mapping_size*2,mapping_size*2,kernel_size=1,padding=0,bias=False),
            nn.PReLU(num_parameters=mapping_size*2, init=0.01),
            nn.GroupNorm(1,mapping_size*2)
            )
        
        self.ffmlayer3 = nn.Sequential(
            nn.Conv2d(mapping_size*2,1,kernel_size=1,padding=0,bias=False),
            nn.PReLU(num_parameters=1, init=0.)
            )
        
        self.prelu = nn.PReLU(num_parameters=1, init=0.01)
        self.d = nn.Dropout(0.15)
        
    def forward(self, x):
        x0 = self.ffm(x)
        x1 = self.ffmlayer1(self.d(x0))
        x2 = self.ffmlayer2(x0+x1)
        x3 = self.ffmlayer3(x1+x2)
        return x3 + x
