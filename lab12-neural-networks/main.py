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
print(net)
