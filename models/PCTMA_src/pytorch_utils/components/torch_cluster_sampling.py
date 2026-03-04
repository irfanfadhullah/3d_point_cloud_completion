import torch
import os
import sys

from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_pointnet
import copy


class FPS:
    def __init__(self, x, ratio):
        self.original_x = copy.deepcopy(x)
        self.x = copy.deepcopy(x)
        self.ratio = ratio
        self.batch_size = 1
        if x.dim() == 3:
            self.batch_size, self.N, self.dim = x.shape
            self.x = self.x.reshape(-1, self.dim)
        else:
            self.N = self.x.shape[0]
            self.dim = self.x.shape[1]

        batch_index = torch.arange(self.batch_size, device=self.x.device)
        self.batch_fps = batch_index.repeat(self.N, 1).transpose(0, 1).reshape(-1)

    def get(self):
        fps_idx = fps_pointnet(self.x.unsqueeze(0), int(self.N * self.ratio)).squeeze(0).long()
        fps_center = self.x[fps_idx].reshape(self.batch_size, -1, self.dim)
        return fps_center, self.x, fps_idx


def farthest_point_sampling(x, ratio):
    if ratio > 1:
        ratio = ratio / x.shape[1]
    
    batch_size = 1
    if x.dim() == 3:
        batch_size, N, dim = x.shape
        x_reshaped = x
    else:
        N = x.shape[0]
        dim = x.shape[1]
        x_reshaped = x.unsqueeze(0)

    fps_idx = fps_pointnet(x_reshaped, int(N * ratio)).long()
    
    if x.dim() == 2:
        fps_idx = fps_idx.squeeze(0)
        fps_center = x[fps_idx].reshape(batch_size, -1, dim)
    else:
        fps_center = torch.gather(x, 1, fps_idx.unsqueeze(-1).expand(-1, -1, dim))
        
    return fps_center, x, fps_idx


def K_NN(points, maxKnn, queryPoints):
    points = points.transpose(2, 1)
    queryPoints = queryPoints.transpose(2, 1)
    dist = torch.cdist(queryPoints, points)
    indices = torch.topk(dist, k=maxKnn, dim=-1, largest=False)[1]
    return indices


class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points,))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()


class BkNN:  # B,N,D
    def __init__(self, points, maxKnn, includeSelf=False, queryPoints=None):
        if not includeSelf:
            self.maxKnn = maxKnn + 1
        else:
            self.maxKnn = maxKnn
        self.points = points
        
        if queryPoints is not None:
            self.queryPoints = queryPoints
            dist = torch.cdist(queryPoints, points)
            self.indices = torch.topk(dist, k=self.maxKnn, dim=-1, largest=False)[1]
            self.cluster = torch.gather(points.unsqueeze(1).expand(-1, queryPoints.shape[1], -1, -1), 
                                        2, self.indices.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
        else:
            dist = torch.cdist(points, points)
            self.indices = torch.topk(dist, k=self.maxKnn, dim=-1, largest=False)[1]
            self.cluster = torch.gather(points.unsqueeze(1).expand(-1, points.shape[1], -1, -1), 
                                        2, self.indices.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))

        if not includeSelf:
            self.remove_self_loop()

    def remove_self_loop(self):
        self.cluster = self.cluster[:, :, 1:, :]
        self.indices = self.indices[:, :, 1:]

    def query(self, index):
        return self.cluster[:, index, :, :]

    def queryIndics(self, index):
        return self.query(index), self.indices[:, index, :]


class BraidusNN:
    def __init__(self):
        pass

    def get(self):
        pass


if __name__ == '__main__':
    x = torch.rand(3, 15, 3)
    my_fps = FPS(x, ratio=0.5)
    fps_center, x, fps_idx = my_fps.get()
