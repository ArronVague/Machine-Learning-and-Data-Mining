import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
import os

#  读入训练数据集'wine_train.csv'与测试数据集'wine_test.csv'。
wine_train = pd.read_csv("wine_train.csv")
wine_test = pd.read_csv("wine_test.csv")


# 利用线性层和激活函数搭建一个神经网络，要求输入和输出维度与数据集维度一致，而神经网络深度、隐藏层大小、激活函数种类等超参数自行调整。
# 输入维度为11
input_dim = wine_train.shape[1] - 1
hidden_dim = 5
output_dim = 1
# print(input_dim)


# 定义神经网络模型，继承自nn.Module
class Net(nn.Module):
    # 输入层的维度为 input_dim
    # 隐藏层的维度为 hidden_dim
    # 输出层的维度为 output_dim
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 激活函数relu，用于在全连接层之间加入非线性变换
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out1 = self.relu(out)
        out2 = self.fc2(out1)
        return out1, out2


# 创建神经网络模型实例并输出
net = Net(input_dim, hidden_dim, output_dim)
# print(net)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


Epoch = 1000

for epoch in range(Epoch):
    print("Epoch:", epoch)
    # 读取训练数据集的特征和标签
    train_features = torch.tensor(wine_train.iloc[:, 0:11].values)
    train_labels = torch.tensor(wine_train.iloc[:, 11].values)
    train_features = train_features.float()
    train_labels = train_labels.float()
    # print(train_features)
    # print(train_labels)
    # print(train_features.shape)
    # print(train_labels.shape)

    # 读取测试数据集的特征和标签
    test_features = torch.tensor(wine_test.iloc[:, 0:11].values)
    test_labels = torch.tensor(wine_test.iloc[:, 11].values)
    test_features = test_features.float()
    test_labels = test_labels.float()
    # print(test_features)
    # print(test_labels)
    # print(test_features.shape)
    # print(test_labels.shape)

    # 进行一次forward()前向传播
    # 这是PyTorch中的一种简便写法，等价于net.forward(input)
    output1, output2 = net(train_features)

    # 计算损失函数
    loss = criterion(output2, train_labels)

    # 清空梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 输出损失函数值
    # print("Loss:", loss.item())

    # 计算测试集上的准确率
    test_output1, test_output2 = net(test_features)
    test_loss = criterion(test_output2, test_labels)
    print("Test loss:", test_loss.item())

    # 计算训练集上的准确率
    train_output1, train_output2 = net(train_features)
    train_loss = criterion(train_output2, train_labels)
    print("Train loss:", train_loss.item())
