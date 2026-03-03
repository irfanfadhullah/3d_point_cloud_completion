import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import random_split
import torchvision
from torchvision import transforms, models


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        # Keep up to layer3 (256 channels) — [:-3] removes layer4, avgpool, fc
        self.base = nn.Sequential(*list(base.children())[:-3])

    def forward(self, x):
        x = self.base(x)  # (B, 256, 14, 14) for 224x224 input
        x = x.view(x.size(0), 256, -1)  # (B, 256, 196)
        return x


if __name__ == '__main__':

    x = torch.randn(2, 3, 224, 224).cuda()
    model = ResNet().cuda()
    out = model(x)
    print('resnet:', out.shape)
    # print(x.shape)                  # resnet: torch.Size([2, 10])
    # summary(model, input_size = (3, 224, 224))
