from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

'''
使用pandas库将训练数据集'train_titanic.csv'载入到Dataframe对象中
'''
df = pd.read_csv("train_titanic.csv")

'''
给定任何标签数组计算其信息熵
输入：标签数组
输出：该数组对应的信息熵
'''
def entropy(label):
    # 使用numpy中的unique实现计数
    unique_labels, label_counts = np.unique(label, return_counts=True)
    total_samples = len(label)
    ent = 0

    for count in label_counts:
        probability = count / total_samples
        # 计算信息熵
        ent -= probability * np.log2(probability)

    return ent

'''
函数功能为将所给的数据集按照指定维度的特征进行划分为若干个不同的数据集
【输入】：特征集合，标签集合，指定维度
【输出】：划分后所得到的子树属性集合，子树标记集合
'''
def split(feature, label, d):
    # 使用numpy中的unique实现非重复值的提取
    unique_values = np.unique(feature[:, d])
    split_feature = []
    split_label = []

    for value in unique_values:
        indices = np.where(feature[:, d] == value)
        split_feature.append(feature[indices])
        split_label.append(label[indices])

    return split_feature, split_label

'''
函数功能为进行【一次】决策树的结点划分，遍历找出该特征集合中信息增益(使用ID3算法中的公式计算)【最大】的特征
输入：特征集合，标签集合
输出：该次划分的最佳信息增益值，最佳划分维度
'''
def one_split_ID3(x_data, y_label):
    num_samples = len(x_data)
    num_features = x_data.shape[1]
    base_entropy = entropy(y_label)
    best_entropy = 0.0
    best_dimension = None

    for feature_dim in range(num_features):
        feature_values = x_data[:, feature_dim]
        split_feature, split_label = split(x_data, y_label, feature_dim)
        new_entropy = 0.0

        for i in range(len(split_feature)):
            sub_feature = split_feature[i]
            sub_label = split_label[i]
            p = len(sub_feature) / num_samples
            new_entropy += p * entropy(sub_label)

        information_gain = base_entropy - new_entropy
        
        # 记录最佳的信息增益值和对应的特征的维数
        if information_gain > best_entropy:
            best_entropy = information_gain
            best_dimension = feature_dim

    return best_entropy, best_dimension

'''
函数功能为进行【一次】决策树的结点划分，遍历找出该特征集合中信息增益率(使用C4.5算法中的公式计算)【最大】的特征
输入：特征集合，标签集合
输出：最佳划分的信息增益率值，对应的划分维度
'''
def one_split_C4_5(x_data, y_label):
    num_features = x_data.shape[1]
    best_entropy = 0.0
    best_dimension = None

    for dimension in range(num_features):
        feature_values = x_data[:, dimension]
        split_feature, split_label = split(x_data, y_label, dimension)
        new_entropy = 0.0
        intrinsic_value = entropy(feature_values)

        for i in range(len(split_feature)):
            sub_features = split_feature[i]
            sub_labels = split_label[i]
            p = len(sub_features) / len(x_data)
            new_entropy += p * entropy(sub_labels)

        information_gain = entropy(y_label) - new_entropy
        gain_ratio = information_gain / intrinsic_value

        # 记录最佳的信息增益率和对应的特征维数
        if gain_ratio > best_entropy:
            best_entropy = gain_ratio
            best_dimension = dimension

    return best_entropy, best_dimension

'''
进行【一次】决策树的结点划分，遍历找出该特征集合中基尼系数(使用CART算法中的公式计算)最小的特征以及最佳的划分值
输入：特征集合，标签集合
输出：最佳的基尼系数，对应的划分维度，最佳划分值
'''
def one_split_CART(x_data, y_label):
    def gini_index(label):
        unique_labels, label_counts = np.unique(label, return_counts=True)
        total_samples = len(label)
        gini = 1

        for count in label_counts:
            probability = count / total_samples
            gini -= probability**2

        return gini

    def split_by_value(feature, label, value):
        indices = np.where(feature == value)
        split_label = label[indices]

        return split_label

    num_features = x_data.shape[1]
    best_entropy = float("inf")
    best_dimension = None
    best_value = None

    for dimension in range(num_features):
        feature_values = x_data[:, dimension]
        unique_values = np.unique(feature_values)

        for value in unique_values:
            split_label = split_by_value(feature_values, y_label, value)
            gini_index_left = gini_index(split_label)
            gini_index_right = gini_index(y_label) - gini_index_left
            total_samples = len(y_label)
            gini_index_dimension = (
                len(split_label) / total_samples
            ) * gini_index_left + (
                len(y_label) - len(split_label)
            ) / total_samples * gini_index_right

            # 记录最小的基尼系数、对应的特征维数和非重复值（分类值）
            if gini_index_dimension < best_entropy:
                best_entropy = gini_index_dimension
                best_dimension = dimension
                best_value = value

    return best_entropy, best_dimension, best_value

# 提取特征列
features = ["Sex", "sibsp", "Parch", "Pclass"]
x_data = df[features].values

# 提取标签列
y_label = df["Survived"].values


'''
应用之前你在第4、5、6个部分编写的三个函数，在训练数据集'train_titanic.csv'上依次使用这些函数进行【一次】结点划分，并输出对应的最佳特征维数以及相应的信息增益值/信息增益率/(基尼系数和分类值)
'''
ID3_best_entropy, ID3_best_dimension = one_split_ID3(x_data, y_label)
print("one_split_ID3:")
print("最佳特征维数:", ID3_best_dimension)
print("最佳信息增益值:", ID3_best_entropy)

C4_5_best_entropy, C4_5_best_dimension = one_split_C4_5(x_data, y_label)
print("one_split_C4_5:")
print("最佳特征维数:", C4_5_best_dimension)
print("最佳信息增益率:", C4_5_best_entropy)

CART_best_entropy, CART_best_dimension, CART_best_value = one_split_CART(
    x_data, y_label
)
print("one_split_CART:")
print("最佳特征维数:", CART_best_dimension)
print("最佳基尼系数:", CART_best_entropy)
print("最佳分类值:", CART_best_value)
