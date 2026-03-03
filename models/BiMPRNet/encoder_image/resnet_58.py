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
        self.base = nn.Sequential(*list(base.children())[:-4])  # 取出ResNet模型的前面部分（去掉最后4个层），作为模型的特征提取器部分
        self.base_1 = nn.Sequential(*list(base.children())[:-3])  # 取出ResNet模型的前面部分（去掉最后3个层），作为模型的特征提取器部分
        in_features = base.fc.in_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        y = self.base_1(x)
        y = y.view(y.size(0), 256, -1)

        x = self.base(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), 128, -1)
        # x = x.view(x.size(0), 256, -1)

        return x.squeeze(), y


if __name__ == '__main__':

    x = torch.randn(2, 3, 224, 224).cuda()
    model = ResNet().cuda()
    out = model(x)
    print('resnet:', out.shape)
    # print(x.shape)                  # resnet: torch.Size([2, 10])
    # summary(model, input_size = (3, 224, 224))
