import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cvxopt
from cvxopt import matrix
from cvxopt import solvers

# # Q = 2 * matrix([[2, 0.5], [0.5, 1]])
# # p = matrix([1.0, 1.0])
# # G = matrix([[-1.0, 0.0], [0.0, -1.0]])
# # h = matrix([0.0, 0.0])
# # A = matrix([1.0, 1.0], (1, 2))
# # b = matrix(1.0)
# # sol = solvers.qp(Q, p, G, h, A, b)
# # print(sol["x"])

dataset2 = pd.read_csv("dataset2.csv")
# 将数据类型转换成np.double
dataset2 = np.array(dataset2, dtype=np.double)
plot_x1 = dataset2[:, 0]
plot_x2 = dataset2[:, 1]
# y为+1的打上红色，-1的打上绿色
plot_y = dataset2[:, 2]
for i in range(len(plot_y)):
    if plot_y[i] == 1:
        plt.scatter(plot_x1[i], plot_x2[i], c="r", marker="o")
    else:
        plt.scatter(plot_x1[i], plot_x2[i], c="g", marker="o")
# plt.show()

X = dataset2[:, 0:2]
# # print(X)
Y = dataset2[:, 2]
X_prime = X * Y.reshape(-1, 1)
# # print(X_prime)
P = matrix(np.dot(X_prime, X_prime.T))
q = matrix(-1 * np.ones(len(X)))
# 创建一个2m*m的矩阵
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
# print(lamda_star)

omega_star = sum(lamda_star[i] * Y[i] * X[i] for i in range(len(X)))
b_star = [
    Y[i] - np.dot(omega_star.T, X[i])
    for i in range(len(X))
    if 1e-5 < lamda_star[i] < C - 1e-5
]
print(b_star)

for i in range(len(plot_y)):
    if 1e-5 < lamda_star[i] < C - 1e-5:
        plt.scatter(plot_x1[i], plot_x2[i], c="b", marker="o")
    elif plot_y[i] == 1:
        plt.scatter(plot_x1[i], plot_x2[i], c="r", marker="o")
    else:
        plt.scatter(plot_x1[i], plot_x2[i], c="g", marker="o")

# 画出决策边界
x1 = np.arange(-4, 12, 0.01)
x2 = np.arange(-4, 10, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y0 = omega_star[0] * x1 + omega_star[1] * x2 + b_star[0]
y1 = omega_star[0] * x1 + omega_star[1] * x2 + b_star[1]
y2 = omega_star[0] * x1 + omega_star[1] * x2 + b_star[2]
plt.contour(x1, x2, y0, [0], colors="red")
# 将间隔边界用虚线画出来
plt.contour(x1, x2, y1, [1], colors="black", linestyles="dashed")
plt.contour(x1, x2, y2, [-1], colors="black", linestyles="dashed")

plt.show()
