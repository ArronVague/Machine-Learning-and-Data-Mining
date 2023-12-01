import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
import os


# # 定义神经网络模型，继承自nn.Module
# class Net(nn.Module):
#     # 输入层的维度为 input_dim
#     # 隐藏层的维度为 hidden_dim
#     # 输出层的维度为 output_dim
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         # 激活函数relu，用于在全连接层之间加入非线性变换
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.fc1(x)
#         out1 = self.relu(out)
#         out2 = self.fc2(out1)
#         return out1, out2


# # 创建神经网络模型实例并输出
# net = Net(10, 5, 1)
# print(net)

# # 该神经网络中可学习的参数可以通过net.parameters()访问
# params = list(net.parameters())
# print([params[i].size() for i in range(len(params))])
# print("Parameters:", params)

# net.eval()
# # 输入维度为10，生成数据
# input = torch.ones([1, 10])
# input = input.float()

# # 进行一次forward()前向传播
# # 这是PyTorch中的一种简便写法，等价于net.forward(input)
# output1, output2 = net(input)

# # 前向传播并输出每一层的输出值
# print("Output of first layer:", output1)
# print("Output of second layer:", output2)


# 创建一个简单的线性回归模型
model = nn.Linear(1, 1)
print(list(model.parameters()))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

Epoch = 10

# 生成数据
inputs = torch.tensor([[1.0], [2.0], [3.0]])
labels = torch.tensor([[2.0], [4.0], [6.0]])

# 模拟训练过程
for epoch in range(Epoch):
    # 模拟输入数据和标签

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 梯度清零
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 打印梯度值
    print(
        "Epoch [{}/{}], Loss: {}".format(epoch + 1, Epoch, loss),
        ". Gradient: {}".format(model.weight.grad),
    )
    # print('Gradient: {}'.format(model.weight.grad))
