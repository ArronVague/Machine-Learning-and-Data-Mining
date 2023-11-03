import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import warnings
import random

warnings.filterwarnings("ignore")
from pandas.core.frame import DataFrame
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel("ERROR")


def DBSCAN(D, epsilon, MinPts):
    # 欧式距离
    def euclidean_distance(x, y):
        return math.sqrt(((x - y) ** 2).sum())

    def get_neighbors(D, index, epsilon):
        neighbors = []
        # 遍历所有样本
        for i in range(len(D)):
            if i == index:
                continue
            # 如果样本与当前样本的距离小于等于邻域半径，则将其加入邻域内
            if euclidean_distance(D[i], D[index]) <= epsilon:
                neighbors.append(i)

        return neighbors

    def core_set(D, epsilon, MinPts):
        # 初始化核心对象集合
        core = set()

        # 对每个样本进行遍历
        for i in range(len(D)):
            # 获取邻域内的所有样本的索引
            neighbors = get_neighbors(D, i, epsilon)

            # 如果邻域内的样本数量大于等于最小样本数，则将当前样本标记为核心对象
            if len(neighbors) >= MinPts:
                core.add(i)

        return list(core)

    # 初始化标签数组，0表示未分类
    labels = [0 for i in range(len(D))]

    # 生成核心对象集合
    core = core_set(D, epsilon, MinPts)
    # print(core)

    # 定义当前簇的标签
    cluster_id = 1

    # 对核心对象集合进行遍历
    for i in range(len(core)):
        # 如果核心对象已经分类，则跳过
        if labels[core[i]] != 0:
            continue

        # 创建一个新的簇，将核心对象标记为该簇
        labels[core[i]] = cluster_id

        # 获取由核心对象密度直达的样本集合Δ
        s = get_neighbors(D, core[i], epsilon)

        # 遍历样本集合Δ
        while s:
            # print(s)

            # 取出一个样本
            t = s.pop()

            # 如果样本已经分类，则跳过
            if labels[t] != 0:
                continue

            # 将样本标记为当前簇
            labels[t] = cluster_id

            # 获取由样本密度直达的样本集合Δ'
            s_prime = get_neighbors(D, t, epsilon)

            # 如果样本是核心对象，则将Δ'中的样本加入Δ
            if t in core:
                for i in range(len(s_prime)):
                    if labels[s_prime[i]] != 0:
                        s.append(s_prime[i])
        # print("yes")
        cluster_id += 1

    # print(labels)

    # 将数据集的二维特征值作为绘图的横纵坐标，将所有样本绘制到一张图中，其中同一聚类的样本点绘制为相同颜色
    plt.figure(figsize=(8, 8))
    plt.title("DBSCAN")
    plt.xlabel("x")
    plt.ylabel("y")
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 0, 0),
        (255, 255, 255),
    ]

    for i in range(len(D)):
        color_index = labels[i] % len(colors)
        # 获取对应的RGB值
        rgb = colors[color_index]
        # 将RGB值转换为0-1范围的浮点数
        normalized_rgb = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        plt.scatter(D[i][0], D[i][1], c=[normalized_rgb], marker="o")
    plt.show()
