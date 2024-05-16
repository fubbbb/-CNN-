import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,sampler,Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import timeit
from PIL import Image
import os
import numpy as np
import scipy.io
import torchvision.models.inception as inception

label_mat=scipy.io.loadmat('F:/pythonProject/Basketball/datasets/q3_2_data.mat')
label_train=label_mat['trLb']
print('train len：',len(label_train))
label_val=label_mat['valLb']
print('val len: ',len(label_val))


class BasketballDataset(Dataset):
    """Action dataset."""

    def __init__(self, root_dir, labels=[], transform=None):
        """
        Args:
            root_dir (string): 整个数据的路径。
            labels(list): 图片的标签。
            transform (callable, optional): 想要对数据进行的处理函数。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.length = len(os.listdir(self.root_dir))
        self.labels = labels

    def __len__(self):  # 该方法只需返回数据的数量。
        return self.length * 3  # 因为每个视频片段都包含3帧。

    def __getitem__(self, idx):  # 该方法需要返回一个数据。

        folder = idx // 3 + 1
        imidx = idx % 3 + 1
        folder = format(folder, '05d')
        imgname = str(imidx) + '.jpg'
        img_path = os.path.join(self.root_dir, folder, imgname)
        image = Image.open(img_path)

        if len(self.labels) != 0:
            Label = self.labels[idx // 3][0] - 1
        if self.transform:  # 如果要先对数据进行预处理，则经过transform函数。
            image = self.transform(image)
        if len(self.labels) != 0:
            sample = {'image': image, 'img_path': img_path, 'Label': Label}
        else:
            sample = {'image': image, 'img_path': img_path}
        return sample


image_dataset=BasketballDataset(root_dir='F:/pythonProject/Basketball/datasets/trainClips/', labels=label_train,transform=T.ToTensor())
# torchvision.transforms中定义了非常多对图像的预处理方法，这里使用的ToTensor方法为将0～255的RGB值映射到0～1的Tensor类型。
for i in range(3):
    sample=image_dataset[i]
    print(sample['image'].shape)
    print(sample['Label'])
    print(sample['img_path'])

image_dataloader = DataLoader(image_dataset, batch_size=4,
                        shuffle=True)

for i, sample in enumerate(image_dataloader):
    sample['image']=sample['image']
    print(i,sample['image'].shape, sample['img_path'],sample['Label'])
    if i > 5:
        break

image_dataset_train=BasketballDataset(root_dir='F:/pythonProject/Basketball/datasets/trainClips/',labels=label_train,transform=T.ToTensor())

image_dataloader_train = DataLoader(image_dataset_train, batch_size=32,
                        shuffle=True)
image_dataset_val=BasketballDataset(root_dir='F:/pythonProject/Basketball/datasets/valClips/',labels=label_val,transform=T.ToTensor())

image_dataloader_val = DataLoader(image_dataset_val, batch_size=32,
                        shuffle=False)
image_dataset_test=BasketballDataset(root_dir='F:/pythonProject/Basketball/datasets/testClips/',labels=[],transform=T.ToTensor())

image_dataloader_test = DataLoader(image_dataset_test, batch_size=32,
                        shuffle=False)
dtype = torch.FloatTensor # 这是pytorch所支持的cpu数据类型中的浮点数类型。

print_every = 100   # 这个参数用于控制loss的打印频率，因为我们需要在训练过程中不断的对loss进行检测。


def reset(m):   # 这是模型参数的初始化
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # 读取各个维度。
        return x.view(N, -1)  # -1代表除了特殊声明过的以外的全部维度。


fixed_model_base = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=7, stride=1), #3*64*64 -> 8*58*58
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 2),    # 8*58*58 -> 8*29*29
                nn.Conv2d(8, 16, kernel_size=7, stride=1), # 8*29*29 -> 16*23*23
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride = 2), # 16*23*23 -> 16*11*11
                Flatten(),
                nn.ReLU(inplace=True),
                nn.Linear(1936, 10)     # 1936 = 16*11*11
            )
# 这里模型base.type()方法是设定模型使用的数据类型，之前设定的cpu的Float类型。
# 如果想要在GPU上训练则需要设定cuda版本的Float类型。
fixed_model = fixed_model_base.type(dtype)

x = torch.randn(32, 3, 64, 64).type(dtype)
x_var = Variable(x.type(dtype)) # 需要将其封装为Variable类型。
ans = fixed_model(x_var)

print(np.array(ans.size())) # 检查模型输出。
np.array_equal(np.array(ans.size()), np.array([32, 10]))