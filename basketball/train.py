import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Basketball.basketball.pre import get_dataloader
from Basketball.basketball.model import Cnn


dtype = torch.FloatTensor

print_every = 100

model = Cnn()
loss_fun = nn.CrossEntropyLoss()
opt_fun = optim.Adam(model.parameters(), lr=0.0001)  # 使用Adam优化器进行优化

def train(epoch):
    dataloader_train = get_dataloader('train')
    dataloader_val = get_dataloader('val')
    model.train()
    for i in range(epoch):
        for idx, sample in enumerate(dataloader_train):
            img = Variable(sample['image'])
            labels = Variable(sample['Label'].long())

            output = model(img)
            loss = loss_fun(output, labels)

            opt_fun.zero_grad()
            loss.backward()
            opt_fun.step()

            if idx % 100 == 0:
                model.eval()
                acc_count = 0.0
                all_count = 0
                with torch.no_grad():
                    for idx, sample in enumerate(dataloader_val):
                        img = Variable(sample['image'])
                        labels = Variable(sample['Label'].long())
                        output = model(img)

                        _, pre = torch.max(output, dim=1)
                        acc_count += (pre == labels).sum().item()
                        all_count += labels.size()[0]

                print('第{}轮，第{}次，loss为: {},验证集准确率为：{}'.format(i + 1, idx + 1, loss.item(), acc_count / all_count))

if __name__ == '__main__':
    train(1)
    torch.save(model.state_dict(), 'F:/pythonProject/Basketball//model/basketball.pth')