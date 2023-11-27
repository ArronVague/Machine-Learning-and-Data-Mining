import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cvxopt
from cvxopt import matrix
from cvxopt import solvers

# Q = 2 * matrix([[2, 0.5], [0.5, 1]])
# p = matrix([1.0, 1.0])
# G = matrix([[-1.0, 0.0], [0.0, -1.0]])
# h = matrix([0.0, 0.0])
# A = matrix([1.0, 1.0], (1, 2))
# b = matrix(1.0)
# sol = solvers.qp(Q, p, G, h, A, b)
# print(sol["x"])

Raisin_train = pd.read_csv("Raisin_train.csv")
# 将数据类型转换成np.double
Raisin_train = np.array(Raisin_train, dtype=np.double)

X = Raisin_train[:, 0:7]
# print(X)
Y = Raisin_train[:, 7]


# print(Y)
# 使用核函数求P矩阵
def K(x, z):
    return np.dot(x.T, z)


P = matrix(0.0, (len(X), len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        P[i, j] = Y[i] * Y[j] * K(X[i], X[j])

# # 硬间隔
# q = matrix(-1 * np.ones(len(X)))
# G = matrix(-1 * np.eye(len(X)))
# h = matrix(np.zeros(len(X)))
# A = matrix(Y.reshape(1, -1))
# b = matrix(0.0)

# 软间隔
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

b_star = [
    Y[j] - sum(lamda_star[i] * Y[i] * K(X[i], X[j]) for i in range(len(X)))
    for j in range(len(X))
    if 1e-5 < lamda_star[j] < C - 1e-5
]
print(b_star)

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
print(acc / len(Raision_test))
