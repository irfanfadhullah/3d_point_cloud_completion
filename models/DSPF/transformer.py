import torch
import torch.nn as nn

from extensions.pointops.functions import pointops


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        center = pointops.fps(xyz, self.num_group)
        idx = pointops.knn(center, xyz, self.group_size)[0]#(b, G, k)
        neighborhood = pointops.index_points(xyz, idx)#[B, S, [K], C]
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

