import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import ConcatDataset


def data_loader(batch_size):
    # 训练集图像预处理：
    # 对测试集进行图像增强可缓解过拟合的情况
    transform_org = transforms.Compose([transforms.ToTensor(),  # 转Tensor
                                        transforms.Normalize(
                                            mean=[0.5,0.5,0.5],
                                            std=[0.5,0.5,0.5])  # 归一化
                                        ])
    transform_nailong = transforms.Compose([transforms.Resize((32,32)),
                                            transforms.ToTensor(),  # 转Tensor
                                            transforms.Normalize(
                                                mean=[0.5,0.5,0.5],
                                                std=[0.5,0.5,0.5])  # 归一化
                                            ])
    # 加载训练集
    trainset_org = torchvision.datasets.CIFAR10(root='./data',train=True,
                                                download=True,transform=transform_org)
    # 加载测试集
    testset_org = torchvision.datasets.CIFAR10(root='./data',train=False,
                                               download=True,transform=transform_org)

    nailong_train_org = torchvision.datasets.ImageFolder(root=os.path.join("data","train"),
                                                         transform=transform_nailong)
    nailong_test_org = torchvision.datasets.ImageFolder(root=os.path.join("data","test"),
                                                        transform=transform_nailong)
    nailong_train_org.targets = [10 for i in range(len(nailong_train_org.samples))]
    nailong_test_org.targets = [10 for i in range(len(nailong_test_org.samples))]
    # 为训练集创建DataLoader对象
    train_org_loader = torch.utils.data.DataLoader(trainset_org,batch_size=batch_size,
                                                   shuffle=False)
    # 为测试集创建DataLoader对象
    test_org_loader = torch.utils.data.DataLoader(testset_org,batch_size=batch_size,
                                                  shuffle=False)
    train_nailong_loader = torch.utils.data.DataLoader(nailong_train_org,batch_size=batch_size,
                                                       shuffle=False)
    test_nailong_loader = torch.utils.data.DataLoader(nailong_test_org,batch_size=batch_size,
                                                      shuffle=False)
    train_nailong_loader.dataset.class_to_idx['nailong'] = 10
    test_nailong_loader.dataset.class_to_idx['nailong'] = 10
    return train_org_loader,test_org_loader,train_nailong_loader,test_nailong_loader

# data_loader(4)
