from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# p代表有毒，e代表无毒
train_df = pd.read_csv("train_mushroom.csv")
test_df = pd.read_csv("test_mushroom.csv")

train = np.array(train_df)
test = np.array(test_df)

# 计算每个标签值y对应的先验概率P(y)
# 𝑃(𝑦)=|𝐷𝑦||𝐷|

# 其中 𝐷𝑦
#  为标签值为y的样本集合， |𝐷𝑦|
#  为这个集合的样本个数；D为所有样本集合，|D|为所有样本个数
Dy = len(train)
label_count = Counter(train[:, -1])
priori_probability = {}
for k, v in label_count.items():
    priori_probability[k] = v / Dy
print("before using laplacian smoothing: ", priori_probability)

# 3) 对于数据集中的每个特征的非重复特征值 𝑥𝑖
#  ，计算给定标签值y时特征值 𝑥𝑖
#  的条件概率 𝑃(𝑥𝑖│𝑦)
#  ,
# 𝑃(𝑥𝑖│𝑦)=|𝐷𝑥𝑖,𝑦||𝐷𝑦|

# 𝐷𝑥𝑖,𝑦
#  为标签值为y，特征值为 𝑥𝑖
#  的样本集合； |𝐷𝑥𝑖,𝑦|
#  为该集合的样本个数

# 首先遍历数据集D中的每个特征，将每个特征的非重复值取出
num_feature = train.shape[1] - 1
feature_unique = [0] * num_feature
for i in range(num_feature):
    feature_unique[i] = set(train[:, i])
# print(feature_unique)

# 根据标签值将数据集D分为两个子数据集，分别包括所有标签值为p的样本和所有标签值为e的样本。
conditional_probability = {}
D = {}
for k in label_count.keys():
    D[k] = train[train[:, -1] == k]


def cal_conditional_probability(D, feature_unique):
    for i in range(num_feature):
        for feature in feature_unique[i]:
            for k in label_count.keys():
                Dxy = D[k][D[k][:, i] == feature]
                conditional_probability[(i, feature, k)] = len(Dxy) / len(D[k])


cal_conditional_probability(D, feature_unique)
# print(conditional_probability)
# print(conditional_probability[(0, "k", "p")])


def pro(a, index):
    res = priori_probability[index]
    for i, x in enumerate(a):
        if (i, x, index) not in conditional_probability.keys():
            return 0
        res *= conditional_probability[(i, x, index)]
    return res


# print(pro(["k", "y", "n", "f", "s", "c", "n", "b", "o", "e", "w", "v", "d"], "e"))
accuracy = 0
for a in test:
    p = pro(a[:-1], "p")
    e = pro(a[:-1], "e")
    predict = "p" if p > e else "e"
    accuracy += 1 if predict == a[-1] else 0

accuracy = accuracy / len(test)
print("before using laplacian smoothing: ", accuracy)

zero = False
for v in conditional_probability.values():
    if v == 0:
        zero = True
        break

# print(zero)
# laplacian_smoothing
priori_probability = {}
for k, v in label_count.items():
    priori_probability[k] = (v + 1) / (Dy + label_count.keys().__len__())
print("after using laplacian smoothing: ", priori_probability)


conditional_probability = {}
D = {}
for k in label_count.keys():
    D[k] = train[train[:, -1] == k]


def cal_conditional_probability(D, feature_unique):
    for i in range(num_feature):
        for feature in feature_unique[i]:
            for k in label_count.keys():
                Dxy = D[k][D[k][:, i] == feature]
                conditional_probability[(i, feature, k)] = (len(Dxy) + 1) / (
                    len(D[k]) + feature_unique[i].__len__()
                )


cal_conditional_probability(D, feature_unique)

accuracy = 0
for a in test:
    p = pro(a[:-1], "p")
    e = pro(a[:-1], "e")
    predict = "p" if p > e else "e"
    accuracy += 1 if predict == a[-1] else 0

accuracy = accuracy / len(test)
print("after using laplacian smoothing: ", accuracy)
