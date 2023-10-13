from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("train_titanic.csv")


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


test_f = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
test_l = np.array([[0], [1], [2]])
test_d = 0

test_sf, test_sl = split(test_f, test_l, test_d)
print(test_sf, test_sl)


def one_split_ID3(x_data, y_label):
    feature_len = len(x_data[0])
    Ent_D = entropy(y_label)
    Gain_D = [0] * feature_len
    for i in range(feature_len):
        split_feature, split_label = split(x_data, y_label, i)
        Gain_D[i] = Ent_D - sum(
            [
                (len(split_label[j]) / len(y_label)) * entropy(split_label[j])
                for j in range(len(split_label))
            ]
        )
    best_entropy = max(Gain_D)
    best_dimension = Gain_D.index(max(Gain_D))
    return best_entropy, best_dimension


def one_split_C4_5(x_data, y_label):
    feature_len = len(x_data[0])
    Gain_ratio = [0] * feature_len
    Ent_D = entropy(y_label)
    Gain_D = [0] * feature_len
    for i in range(feature_len):
        split_feature, split_label = split(x_data, y_label, i)
        Gain_D[i] = Ent_D - sum(
            [
                (len(split_label[j]) / len(y_label)) * entropy(split_label[j])
                for j in range(len(split_label))
            ]
        )
        Gain_ratio[i] = Gain_D[i] / entropy(x_data[:, i])
    best_entropy = max(Gain_ratio)
    best_dimension = Gain_ratio.index(max(Gain_ratio))
    return best_entropy, best_dimension


# data = np.array(df.iloc[:, :-1])
# label = np.array(df.iloc[:, -1])
# label = label.reshape(len(label), 1)
# print(data)
# print(label)
# print(entropy(label))
# one_split_ID3(data, label)
# def one_split_CART(x_data, y_label):

#     return best_entropy, best_dimension,best_value
