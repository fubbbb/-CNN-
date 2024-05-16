from torch.utils.data import DataLoader
import torchvision.transforms as T
import scipy.io

from Basketball.basketball.basketballDataset import BasketballDataset


def get_dataloader(name):

    # 保存的label的标签
    label_mat = scipy.io.loadmat('F:/pythonProject/Basketball/datasets/q3_2_data.mat')
    label_train = label_mat['trLb']
    print('train_len:', len(label_train))
    label_val = label_mat['valLb']
    print('val_len:', len(label_val))

    if name == 'train':
        dataset_train = BasketballDataset(root_dir='F:/pythonProject/Basketball/datasets/trainClips',
                                          labels=label_train,
                                          transform=T.ToTensor())

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=32,
                                      shuffle=True,
                                      num_workers=4)
        return dataloader_train

    if name == 'val':
        dataset_val = BasketballDataset(root_dir='F:/pythonProject/Basketball/datasets/valClips',
                                        labels=label_val,
                                        transform=T.ToTensor())

        dataloader_val = DataLoader(dataset_val,
                                    batch_size=32,
                                    shuffle=True,
                                    num_workers=4)
        return dataloader_val
    if name == 'test':
        dataset_test = BasketballDataset(root_dir='F:/pythonProject/Basketball/datasets/testClips',
                                         transform=T.ToTensor())

        dataloader_test = DataLoader(dataset_test,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=4)
        return dataloader_test


if __name__ == '__main__':

    # 保存的label的标签
    label_mat = scipy.io.loadmat('F:/pythonProject/Basketball/datasets/q3_2_data.mat')
    label_train = label_mat['trLb']
    print('train_len:', len(label_train))
    label_val = label_mat['valLb']
    print('val_len:', len(label_val))

    # Dataloader类所加载的数据必须是pytorch中定义好的Dataset类，
    # 所以我们的第一步，就是将我们的数据封装成一个Dataset类。
    dataset = BasketballDataset(root_dir='F:/pythonProject/Basketball/datasets/trainClips',
                                labels=label_train,
                                transform=T.ToTensor())

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=4)

    for i in range(3):
        sample = dataset[i]
        print(sample['image'].shape)
        print(sample['Label'])
        print(sample['img_path'])

    for i, sample in enumerate(dataloader):
        print(i, sample['image'].shape, sample['img_path'], sample['Label'])
        if i > 5:
            break

