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

from Basketball.basketball.model import Cnn
from Basketball.basketball.pre import get_dataloader


def predict():
    dataloader_test = get_dataloader('test')
    model = Cnn()
    model.load_state_dict(torch.load('F:/pythonProject/Basketball/model/basketball.pth'))

    write_csv = open('F:/pythonProject/Basketball/result.csv', 'w')
    count = 0
    write_csv.write('Id' + ',' + 'Class' + '\n')

    model.eval()
    for t, sample, in enumerate(dataloader_test):
        img = Variable(sample['image'])
        output = model(img)
        _, pre = torch.max(output, dim=1)
        for i in range(len(pre)):
            write_csv.write(str(count) + ',' + str(pre[i].item()) + '\n')
            count += 1

    write_csv.close()
    return count


if __name__ == '__main__':
    count = predict()
    print(count)

