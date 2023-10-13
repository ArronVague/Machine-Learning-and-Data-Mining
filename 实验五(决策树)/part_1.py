from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def entropy(label):
    unique_labels, label_counts = np.unique(label, return_counts=True)
    total_samples = len(label)
    ent = 0

    for count in label_counts:
        probability = count / total_samples
        ent -= probability * np.log2(probability)

    return ent


# arr = np.array([[0], [1]])
# print(entropy(arr))


def split(feature, label, d):
    unique_values = np.unique(feature[:, d])
    split_feature = []
    split_label = []

    for value in unique_values:
        indices = np.where(feature[:, d] == value)
        split_feature.append(feature[indices])
        split_label.append(label[indices])

    return split_feature, split_label


# test_f = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
# test_l = np.array([[0], [1], [2]])
# test_d = 0

# test_sf, test_sl = split(test_f, test_l, test_d)
# print(test_sf, test_sl)


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

        if information_gain > best_entropy:
            best_entropy = information_gain
            best_dimension = feature_dim

    return best_entropy, best_dimension


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

        if gain_ratio > best_entropy:
            best_entropy = gain_ratio
            best_dimension = dimension

    return best_entropy, best_dimension


# x_data = np.array([[0, "A"], [1, "B"], [1, "B"], [0, "A"], [1, "A"]])
# y_label = np.array(["Y", "N", "N", "Y", "Y"])

# # 进行一次决策树节点划分并找出信息增益率最大的特征维度
# best_gain_ratio, best_dimension = one_split_ID3(x_data, y_label)

# print("最佳信息增益率:", best_gain_ratio)
# print("最佳划分维度:", best_dimension)


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

            if gini_index_dimension < best_entropy:
                best_entropy = gini_index_dimension
                best_dimension = dimension
                best_value = value

    return best_entropy, best_dimension, best_value


df = pd.read_csv("train_titanic.csv")


features = ["Sex", "sibsp", "Parch", "Pclass"]
x_data = df[features].values

# 提取标签列
y_label = df["Survived"].values
# print(x_data)

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
