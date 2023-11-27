import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cvxopt
from cvxopt import matrix
from cvxopt import solvers

# 读入数据集并转换成np.double类型，画出数据集的散点图
dataset2 = pd.read_csv("dataset2.csv")
dataset2 = np.array(dataset2, dtype=np.double)
plot_x1 = dataset2[:, 0]
plot_x2 = dataset2[:, 1]
plot_y = dataset2[:, 2]
for i in range(len(plot_y)):
    if plot_y[i] == 1:
        plt.scatter(plot_x1[i], plot_x2[i], c="r", marker="o")
    else:
        plt.scatter(plot_x1[i], plot_x2[i], c="g", marker="o")
plt.show()

# 求出二次规划问题中的P，q，G，h，A，b矩阵，并设置参数c=1
X = dataset2[:, 0:2]
Y = dataset2[:, 2]
X_prime = X * Y.reshape(-1, 1)
P = matrix(np.dot(X_prime, X_prime.T))
q = matrix(-1 * np.ones(len(X)))
G = matrix(np.zeros((2 * len(X), len(X))))
for i in range(len(X)):
    G[i, i] = -1
for i in range(len(X)):
    G[i + len(X), i] = 1
h = matrix(np.zeros(2 * len(X)))
C = 2.0
for i in range(len(X)):
    h[i + len(X)] = C
A = matrix(Y.reshape(1, -1))
b = matrix(0.0)
sol = solvers.qp(P, q, G, h, A, b)
lamda_star = np.array(sol["x"])

# 求出omega_star和b_star，设置阈值threshold=1e-5，筛去非常靠近0和C的分量
threshold = 1e-5
omega_star = sum(lamda_star[i] * Y[i] * X[i] for i in range(len(X)))
b_star = [
    Y[i] - np.dot(omega_star.T, X[i])
    for i in range(len(X))
    if threshold < lamda_star[i] < C - threshold
]

# 画出数据集的散点图、决策边界和间隔边界
for i in range(len(plot_y)):
    if threshold < lamda_star[i] < C - threshold:
        plt.scatter(plot_x1[i], plot_x2[i], c="b", marker="o")
    elif plot_y[i] == 1:
        plt.scatter(plot_x1[i], plot_x2[i], c="r", marker="o")
    else:
        plt.scatter(plot_x1[i], plot_x2[i], c="g", marker="o")
x1 = np.arange(-4, 12, 0.01)
x2 = np.arange(-4, 10, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y0 = omega_star[0] * x1 + omega_star[1] * x2 + b_star[0]
y1 = omega_star[0] * x1 + omega_star[1] * x2 + b_star[1]
y2 = omega_star[0] * x1 + omega_star[1] * x2 + b_star[2]
plt.contour(x1, x2, y0, [0], colors="red")
plt.contour(x1, x2, y1, [1], colors="black", linestyles="dashed")
plt.contour(x1, x2, y2, [-1], colors="black", linestyles="dashed")
plt.show()
