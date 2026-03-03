import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [B,N,K]
    return idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return idx.int()

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.5):#
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.relu=nn.ReLU()
    def forward(self, x):
        B, N, k, C = x.shape
        #print(B, N, k, C)
        qkv = self.qkv(x).reshape(B, N, k, 3, self.num_heads, C // self.num_heads).permute(3,0,4,1,2,5) #[3, B, h, ,N ,k, dim]

        #print('qkv',qkv.shape)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ V).transpose(1, 2).reshape(B, N, k, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #x=self.relu(x)
        return x


class EdgeAttnFeature(nn.Module):
    '''
    Input:
        x:[B,C,N]
    Output:
        feature after Edge_attn: [B, 2*C, N, k]
    '''
    def __init__(self,dim, k=16, idx=None,If_attn=False):
        super().__init__()
        self.k=k
        self.idx=idx
        self.if_attn=If_attn
        if If_attn:
            self.attn=Attention(dim, num_heads=8, qkv_bias=False, qk_scale=None,attn_drop=0.,  proj_drop=0.)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        with torch.no_grad():
            if self.idx is None:
                idx = knn(x, k=self.k)  # [batch_size, num_points, k]
            idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
            idx = idx + idx_base
            idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2,1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, self.k, 1)
        feature=feature - x
        if self.if_attn:
            feature_attn=self.attn(feature) # [B,N,k,C]
            feature = torch.cat((feature_attn, x), dim=3).permute(0, 3, 1, 2).contiguous()#[B,C,N,k]
        else:
            feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2).contiguous()
        #print('feature',feature.shape)
        return feature

class Mlp(nn.Module):
    def __init__(self, in_channel, layer_dims):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            #if ln:
            #    layers.append(nn.LayerNorm(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)
    def forward(self, inputs):
        x=self.mlp(inputs)
        return x

class Mlp_Res(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,if_norm,if_shortcut):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu=nn.ReLU()
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)
        self.if_norm=if_norm
        self.if_shortcut = if_shortcut

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_1(x)
        if self.if_norm == True:
            out = self.norm1(out)
        out =self.relu(out)
        if self.if_shortcut == True:
            out = self.conv_2(torch.relu(out)) + shortcut
        else:
            out = self.conv_2(torch.relu(out))
        return out


def distance_squre(p1,p2):
    tensor=torch.FloatTensor(p1)-torch.FloatTensor(p2)
    val=tensor.mul(tensor)
    val = val.sum()
    return val


class Mlp_Point(nn.Module):
    def __init__(self, PointG_chans,hid_chans_Mlp,out_chans):
        super().__init__()
        self.mlp_1 = Mlp_Res(in_dim=PointG_chans+256, hidden_dim=PointG_chans, out_dim=PointG_chans)
        self.mlp_2 = Mlp_Res(in_dim=PointG_chans, hidden_dim=PointG_chans//2, out_dim=PointG_chans)
        self.mlp_3 = Mlp_Res(in_dim=PointG_chans+256, hidden_dim=PointG_chans, out_dim=out_chans)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x,fea):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1=x
        #print(x1.shape)
        c=torch.cat([x1, fea.repeat((1, 1, x1.size(2)))], 1)
        x1 = self.mlp_1(c)#[B, 640, 256]->
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, fea.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        #completion = self.mlp_4(x3)  # (b, 3, 256)
        completion=x3
        return completion


if __name__=='__main__':
    xyz=torch.randn(10,512,20).cuda()
    Edge=EdgeAttnFeature(dim=512, k=16, idx=None).to('cuda')
    fea=Edge(xyz)
    print('fea',fea.shape)