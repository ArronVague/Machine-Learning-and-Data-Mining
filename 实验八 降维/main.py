import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import warnings
from K_means import K_means
from DBSCAN import DBSCAN

warnings.filterwarnings("ignore")
from pandas.core.frame import DataFrame

df = pd.read_csv("train_data.csv")

# 训练集共有167个样本，每个样本有9个特征值，将原始数据按列组成9行167列的矩阵X
X = np.array(df.iloc[:, :].T)

# 对所有样本进行中心化，即将X的每一行减去这一行的均值
for i in range(X.shape[0]):
    X[i, :] = X[i, :] - np.mean(X[i, :])

# print(X)

# 求出协方差矩阵
C = np.dot(X, X.T) / (X.shape[1] - 1)

# print(C)
# 对协方差矩阵 𝑋
#  . 𝑋𝑇
#  做特征值分解，即求出协方差矩阵的特征值 𝜆⃗ ={𝜆1,𝜆2,...,𝜆𝑑}
#  及对应的特征向量 𝜔⃗ ={𝜔1,𝜔2,...,𝜔𝑑}
#  . 其中 𝜆𝑖∼𝜔𝑖
#  .

lambdas, omegas = np.linalg.eig(C)
# print("该矩阵的特征值：", lambdas)
# print("该矩阵的特征向量：", omegas)

t = 0.99

lambdas_omegas = [(lambdas[i], omegas[:, i]) for i in range(len(lambdas))]
lambdas_omegas = sorted(lambdas_omegas, reverse=True)
# print(lambdas_omegas)

k = len(lambdas_omegas)
for i in range(k + 1):
    if (
        sum([lambdas_omegas[j][0] for j in range(i)])
        / sum([lambdas_omegas[j][0] for j in range(k)])
        >= t
    ):
        k = i
        break

# print(k)

# 将特征向量按对应特征值大小从上到下按行排列，取前k个对应特征值最大的特征向量组成投影矩阵W=( 𝜔1,𝜔2,...,𝜔𝑘
#  )
W = np.array([lambdas_omegas[i][1] for i in range(k)]).T

# print(W)

# 根据公式 𝑌=𝑃.𝑋
#  得到降维到k维后的数据集Y
Y = np.dot(W.T, X)
# print(Y.shape[0])

# K-means聚类
K_means(Y.T, 5)

DBSCAN(Y.T, 35000, 40)
# 35000, 70/50/40看起来不错
