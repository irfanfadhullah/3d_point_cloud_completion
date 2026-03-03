import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
from datetime import datetime, timedelta
from decoder.dec_net import Decoder_Network
from encoder_dgcnn.dgcnn1 import DGCNN
from encoder_image.resnet import ResNet
from config import params

p = params()


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.pos_mlp = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention.transpose(1, 2)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class PCN_Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.latent_dim = latent_dim

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Conv1d(512, 256, 1)

    def forward(self, xyz):
        B, N, _ = xyz.shape

        feature = self.first_conv(xyz.transpose(2, 1))  # (B, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B, 512, N)
        feature = self.second_conv(feature)  # (B, 256, N)

        return feature


class Network1(nn.Module):
    def __init__(self):
        super(Network1, self).__init__()

        # Encoders for images and Point clouds
        self.pc_encoder = DGCNN()
        self.im_encoder = ResNet()

        # Attention layers to fuse the information from the two modalities
        self.cross_attn1 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(p.d_attn)

        self.self_attn2 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(p.d_attn)

        self.cross_attn3 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(p.d_attn)

        self.self_attn4 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm4 = nn.LayerNorm(p.d_attn)

        self.cross_attn5 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm5 = nn.LayerNorm(p.d_attn)

        self.cross_attn6 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm6 = nn.LayerNorm(p.d_attn)

        self.self_attn7 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm7 = nn.LayerNorm(p.d_attn)

        self.cross_attn8 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm8 = nn.LayerNorm(p.d_attn)

        self.self_attn9 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm9 = nn.LayerNorm(p.d_attn)

        self.cross_attn10 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm10 = nn.LayerNorm(p.d_attn)

        self.cs_encoder = PCN_Encoder()

        # Decoder Network to reconstruct the point cloud
        self.decoder = Decoder_Network()

    def forward(self, x_part, view):

        pc_feat, coarse = self.pc_encoder(x_part)  # B x C x N
        im_feat = self.im_encoder(view)  # B x C x N

        coarse = coarse.permute(0, 2, 1)  # B N C
        coarse_feat = self.cs_encoder(coarse)  # B C N

        im_feat = im_feat.permute(0, 2, 1)  # B N C
        pc_feat = pc_feat.permute(0, 2, 1)  # B N C
        coarse_feat = coarse_feat.permute(0, 2, 1)  # B N C

        # x, _ = self.cross_attn1(pc_feat, coarse_feat, coarse_feat)
        # pc_feat = self.layer_norm6(x + pc_feat)  # B x N x F
        #
        # x, _ = self.self_attn1(pc_feat, pc_feat, pc_feat)
        # pc_feat = self.layer_norm7(x + pc_feat)
        # pc_skip = pc_feat
        #
        # x, _ = self.cross_attn2(pc_feat, coarse_feat, coarse_feat)
        # pc_feat = self.layer_norm8(x + pc_feat)
        #
        # x, _ = self.self_attn2(pc_feat, pc_feat, pc_feat)
        # pc_feat = self.layer_norm9(x + pc_feat)
        #
        # x, _ = self.cross_attn3(pc_feat, pc_skip, pc_skip)
        # pc_feat = self.layer_norm10(x + pc_feat)

        x, _ = self.cross_attn1(pc_feat, im_feat, im_feat)
        pc_feat = self.layer_norm1(x + pc_feat)  # B x N x F

        x, _ = self.self_attn2(pc_feat, pc_feat, pc_feat)
        pc_feat = self.layer_norm2(x + pc_feat)
        pc_skip = pc_feat

        x, _ = self.cross_attn6(pc_feat, coarse_feat, coarse_feat)
        pc_feat = self.layer_norm6(x + pc_feat)

        x, _ = self.cross_attn3(pc_feat, im_feat, im_feat)
        pc_feat = self.layer_norm3(x + pc_feat)

        x, _ = self.self_attn4(pc_feat, pc_feat, pc_feat)
        pc_feat = self.layer_norm4(x + pc_feat)

        x, _ = self.cross_attn8(pc_feat, coarse_feat, coarse_feat)
        pc_feat = self.layer_norm8(x + pc_feat)

        x, _ = self.cross_attn5(pc_feat, pc_skip, pc_skip)
        pc_feat = self.layer_norm5(x + pc_feat)

        x_part = x_part.permute(0, 2, 1)  # B x 3 x N ----> B x N x 3

        final = self.decoder(pc_feat, x_part)

        return final


if __name__ == '__main__':
    x_part = torch.randn(16, 3, 2048).cuda()
    view = torch.randn(16, 3, 224, 224).cuda()
    model = Network1().cuda()
    out = model(x_part, view)
    print(out.shape) 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")
    