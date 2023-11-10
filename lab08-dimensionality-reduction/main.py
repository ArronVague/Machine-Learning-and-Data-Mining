import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import warnings

# 将实验七的两种降维算法模块化，用于最后一步的聚类
from K_means import K_means
from DBSCAN import DBSCAN

warnings.filterwarnings("ignore")
from pandas.core.frame import DataFrame

df = pd.read_csv("train_data.csv")

# 转置，将原始数据按列组成9行167列的矩阵
X = np.array(df.iloc[:, :].T)

# 中心化
for i in range(X.shape[0]):
    X[i, :] = X[i, :] - np.mean(X[i, :])

# 协方差矩阵
C = np.dot(X, X.T) / (X.shape[1] - 1)

# 特征值分解
lambdas, omegas = np.linalg.eig(C)

# 这里取阈值为t% = 0.99
t = 0.99

# 根据特征值降序排序，计算降维后的维度k
lambdas_omegas = [(lambdas[i], omegas[:, i]) for i in range(len(lambdas))]
lambdas_omegas = sorted(lambdas_omegas, reverse=True)

k = len(lambdas_omegas)
for i in range(k + 1):
    if (
        sum([lambdas_omegas[j][0] for j in range(i)])
        / sum([lambdas_omegas[j][0] for j in range(k)])
        >= t
    ):
        k = i
        break

# 取前k个特征值对应的特征向量组成矩阵W
W = np.array([lambdas_omegas[i][1] for i in range(k)]).T
bad_W = np.array([lambdas_omegas[i][1] for i in range(-1, -k - 1, -1)]).T
# print(bad_W)

# 计算降到k维后的数据
Y = np.dot(W.T, X)
bad_Y = np.dot(bad_W.T, X)

# K-means聚类
# 参数：降维后的数据，聚类簇数
K_means(Y.T, 3)
K_means(bad_Y.T, 3)

# DBSCAN聚类
# 参数：降维后的数据，半径，最小样本数
DBSCAN(Y.T, 35000, 40)
DBSCAN(bad_Y.T, 35000, 40)

# 35000, 70/50/40看起来不错
