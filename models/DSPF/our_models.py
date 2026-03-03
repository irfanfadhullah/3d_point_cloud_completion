import os
import torch
import copy
import math
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from torch import einsum

from easydict import EasyDict as edict

from models.DSPF.transformer import Group
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.pointops.functions import pointops
from timm.models.layers import trunc_normal_

class MlpConv(nn.Module):
    def __init__(self, input_channel, channels, activation_function=None):
        super(MlpConv, self).__init__()
        self.layer_num = len(channels)
        self.net = nn.Sequential()
        last_channel = input_channel
        for i, channel in enumerate(channels):
            self.net.add_module('Conv1d_%d' % i, nn.Conv1d(last_channel, channel, kernel_size=1))
            if i != self.layer_num - 1:
                self.net.add_module('ReLU_%d' % i, nn.ReLU())
            last_channel = channel
        if activation_function != None:
            self.net.add_module('af', activation_function)

    def forward(self, x):
        return self.net(x)



class Encoder(nn.Module):
    def __init__(self, feat_dim):
        """
        PCN based encoder
        """
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feat_dim, 1)
        )

    def forward(self, x):
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 1024
        return feature_global

    def mean_forward(self, x):
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.mean(feature, dim=2, keepdim=False) # B 1024
        return feature_global


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, num_output=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_output = num_output

        self.mlp1 = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_output)
        )

    def forward(self, z):
        bs = z.size(0)

        pcd = self.mlp1(z).reshape(bs, -1, 3)  #  B M C(3)

        return pcd

class our_model(nn.Module):
    def __init__(self, ):
        super().__init__()
        # self.n_point = cfg.n_point#输出点
        self.feat_dim = 1024

        self.encoder = Encoder(self.feat_dim)
        self.generator = Decoder()



    def forward(self, xyz):
        feat = self.encoder(xyz)
        pred = self.generator(feat).contiguous()
        return pred




