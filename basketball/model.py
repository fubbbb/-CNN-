import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler, Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import timeit
from PIL import Image
import os
import numpy as np
import scipy.io
import torchvision.models.inception as inception


class Cnn(nn.Module):
    def __init__(self, channel=3):
        super(Cnn, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(channel, 8, kernel_size=7, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 16, 7, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(16 * 11 * 11, 10)
        )

    def forward(self, x):
        return self.sequential(x)


if __name__ == '__main__':
    model = Cnn()
    model = model
    x = torch.randn(32, 3, 64, 64)
    x_var = Variable(x)  # 需要将其封装为Variable类型。
    outputs = model(x_var)
    print(np.array(outputs.size()))  # 检查模型输出。
    np.array_equal(np.array(outputs.size()), np.array([32, 10]))

