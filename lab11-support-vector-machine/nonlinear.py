import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cvxopt
from cvxopt import matrix
from cvxopt import solvers

# 读入数据集并转换成np.double类型，画出数据集的散点图
Raisin_train = pd.read_csv("Raisin_train.csv")
Raisin_train = np.array(Raisin_train, dtype=np.double)

X = Raisin_train[:, 0:7]
Y = Raisin_train[:, 7]


# 求出二次规划问题中的P，q，G，h，A，b矩阵，并设置参数c=1
def K(x, z):
    return np.dot(x.T, z)


P = matrix(0.0, (len(X), len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        P[i, j] = Y[i] * Y[j] * K(X[i], X[j])

# 软间隔
q = matrix(-1 * np.ones(len(X)))
G = matrix(np.zeros((2 * len(X), len(X))))
for i in range(len(X)):
    G[i, i] = -1
for i in range(len(X)):
    G[i + len(X), i] = 1
h = matrix(np.zeros(2 * len(X)))
C = 1
for i in range(len(X)):
    h[i + len(X)] = C
A = matrix(Y.reshape(1, -1))
b = matrix(0.0)
sol = solvers.qp(P, q, G, h, A, b)
lamda_star = np.array(sol["x"])

# 求出b_star，设置阈值threshold=1e-5，筛去非常靠近0的分量
threshold = 1e-5
b_star = [
    Y[j] - sum(lamda_star[i] * Y[i] * K(X[i], X[j]) for i in range(len(X)))
    for j in range(len(X))
    if threshold < lamda_star[j] < C - threshold
]

# 读入测试集，用分类决策函数进行预测，输出预测准确率
Raision_test = pd.read_csv("Raisin_test.csv")
Raision_test = np.array(Raision_test, dtype=np.double)


def f(x):
    return np.sign(
        sum(lamda_star[i] * Y[i] * K(X[i], x) for i in range(len(X))) + b_star[0]
    )


acc = 0
for i in range(len(Raision_test)):
    if f(Raision_test[i, 0:7]) == Raision_test[i, 7]:
        acc += 1
print("C：", C)
print("预测准确率：", acc / len(Raision_test))
