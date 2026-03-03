#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Le Wang


import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from models.FDANet.Model_utils import EdgeAttnFeature,Mlp,Mlp_Res
import torch.nn.functional as F

class Convlayer(nn.Module):
    def __init__(self,point_scales):
        super(Convlayer,self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128),2)
        x_256 = torch.squeeze(self.maxpool(x_256),2)
        x_512 = torch.squeeze(self.maxpool(x_512),2)
        x_1024 = torch.squeeze(self.maxpool(x_1024),2)
        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1)
        return x

class FeatureExtract(nn.Module):
    '''
    Input: x:[B,3,N]
    Output: Feature_Extract:[]
    '''
    def __init__(self, N_x,k,in_chans):
        super().__init__()
        self.N_x = N_x
        self.k = k
        self.Edge_attn_feature_0 = EdgeAttnFeature(dim=in_chans[0], k=self.k, idx=None, If_attn=False)
        self.Edge_attn_feature_1 = EdgeAttnFeature(dim=in_chans[1], k=self.k, idx=None, If_attn=False)
        self.Edge_attn_feature_2 = EdgeAttnFeature(dim=in_chans[2], k=self.k, idx=None, If_attn=False)
        self.Edge_attn_feature_3 = EdgeAttnFeature(dim=in_chans[3], k=self.k, idx=None, If_attn=False)
        self.in_chans=in_chans
        self.layer_0 = nn.Sequential(
            nn.Conv2d(in_chans[0]*2, in_chans[0]*2, kernel_size=1, bias=False),
            nn.GroupNorm(4, in_chans[0]*2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_chans[1]*2, in_chans[1]*2, kernel_size=1, bias=False),
            nn.GroupNorm(4, in_chans[1]*2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_chans[2]*2, in_chans[2]*2, kernel_size=1, bias=False),
            nn.GroupNorm(4, in_chans[2]*2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_chans[3] * 2, in_chans[3] * 2, kernel_size=1, bias=False),
            nn.GroupNorm(4, in_chans[3] * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.get_Fea0 = nn.Sequential(
            self.Edge_attn_feature_0,
            self.layer_0,
        )

        self.get_Fea1 = nn.Sequential(
            self.Edge_attn_feature_1,
            self.layer_1,
        )

        self.get_Fea2 = nn.Sequential(
            self.Edge_attn_feature_2,
            self.layer_2,
        )

        self.get_Fea3 = nn.Sequential(
            self.Edge_attn_feature_3,
            self.layer_3,
        )
        self.mlp = nn.Sequential(
            nn.Conv1d(sum(self.in_chans[:])*2, 1920, 1, 1),
            nn.GroupNorm(32, 1920),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1920, 1920, 1)
        )

        self.Convlayers1 = Convlayer(point_scales=self.N_x[1])
        self.Convlayers2 = Convlayer(point_scales=self.N_x[2])

        self.relu=nn.ReLU()
        self.gn = nn.GroupNorm(4,in_chans[0])

        self.mlp_glbfea=nn.Sequential(
            nn.Conv1d(3, 1, 1),
            nn.GroupNorm(1,1),
            nn.ReLU()
        )
        self.code=nn.Sequential(
            nn.Conv1d(15,192,1),
            #nn.GroupNorm(4, 1),
            nn.ReLU()
        )
    def forward(self, x):
        bs=x.shape[0]
        x0 = x
        x1 = gather_operation(x, furthest_point_sample(x.permute(0, 2, 1).contiguous(), self.N_x[1])).permute(0, 2, 1).contiguous()
        x2 = gather_operation(x, furthest_point_sample(x.permute(0, 2, 1).contiguous(), self.N_x[2])).permute(0, 2, 1).contiguous()

        Conv0 = nn.Conv1d(3, self.in_chans[0], 1).to('cuda')
        Fea0_x0 =  self.relu(self.gn(Conv0(x0)))
        Fea1_x0 =  self.relu(self.get_Fea0(Fea0_x0).max(dim=-1, keepdim=False)[0])
        #print('Fea1_x0', Fea1_x0.shape)


        Fea2_x0 =  self.relu(self.get_Fea1(Fea1_x0).max(dim=-1, keepdim=False)[0])
        #print('Fea2_x0', Fea2_x0.shape)


        Fea3_x0 = (self.get_Fea2(Fea2_x0).max(dim=-1, keepdim=False)[0])
        #print('Fea3_x0', Fea3_x0.shape)


        Fea4_x0 = (self.get_Fea3(Fea3_x0).max(dim=-1, keepdim=False)[0])
        #print('Fea4_x0', Fea4_x0.shape)

        Fea_global0 = torch.cat([Fea1_x0, Fea2_x0,Fea3_x0,Fea4_x0], 1)
        #print('Fea_global0', Fea_global0.shape)
        Fea_mlp0 = self.mlp(Fea_global0)  # [B,C,N]

        global_feature0 = torch.max(Fea_mlp0, dim=-1, keepdim=True)[0]
        global_feature1=self.Convlayers1(x1)
        global_feature2=self.Convlayers2(x2)

        global_feature=torch.cat([global_feature0,global_feature1,global_feature2],2).transpose(1,2)
        #print('global_feature0',global_feature.shape)
        global_feature=self.mlp_glbfea(global_feature)
        #print('global_feature1', global_feature.shape)
        global_feature=torch.squeeze( global_feature,1)
        #print('global_feature2',global_feature.shape)
        global_code=global_feature.reshape(bs,-1,128)
        global_code=self.code(global_code)
        #print('global_code', global_code.shape)
        return global_code,global_feature # [B,C,N]

if __name__=='__main__':
    a= torch.randn(4,3,2048).cuda()
    _,_,N=a.size()
    N_x=[N,N//4,N//8]
    k=8
    model = FeatureExtract(N_x,k,in_chans=[32,64,128,256]).to('cuda')
    global_code,global_feature= model(a)

