import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import warnings
import random

warnings.filterwarnings("ignore")

df = pd.read_csv("illness.csv")
df["class"] = df["class"].map({"Abnormal": 0, "Normal": 1}).fillna(-1)

# print(df.isnull().sum())

features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

test_ratio = 0.3

test_samples_per_class = (labels.value_counts() * test_ratio).astype(int)

train_data = pd.DataFrame()
test_data = pd.DataFrame()

for label, count in test_samples_per_class.items():
    class_samples = df[df["class"] == label].sample(n=count, random_state=42)
    test_data = test_data.append(class_samples)
    train_data = train_data.append(df[df["class"] == label].drop(class_samples.index))

train_features = train_data.iloc[:, :-1]
train_labels = train_data.iloc[:, -1]
test_features = test_data.iloc[:, :-1]
test_labels = test_data.iloc[:, -1]


# 批量梯度下降
def batch_gradient_descent(omega_init, features, labels, learning_rate, threshold):
    def omega_update(omega):
        row_len = len(features)
        col_len = len(features[0])
        while True:
            omega_new = omega.copy()
            total_list = [0] * row_len
            for i in range(row_len):
                wx = 0
                for o, x in zip(omega, features[i]):
                    wx += o * x
                ewx = np.exp(wx)
                # print(ewx)
                total_list[i] = ewx / (1 + ewx) - labels[i]
            # print(total_list)
            for i in range(col_len):
                s = 0
                for j, total in enumerate(total_list):
                    s += total * features[j][i]
                # print(s)
                # print(omega_new[i] - learning_rate * s / len(sample))
                omega_new[i] = omega_new[i] - learning_rate * s / row_len
            #     print(omega_new[i])
            # print(omega_new)
            if all(abs(x - y) < threshold for x, y in zip(omega_new, omega)):
                break
            omega = omega_new
        return omega

    omega = omega_update(omega_init)
    return omega


train_features.insert(0, "x0", 1)
test_features.insert(0, "x0", 1)

train_features_array = np.array(train_features)
train_labels_array = np.array(train_labels)

n = len(train_features.columns)
print(n)

omega_init = [1] * n
learning_rate = 0.01
threshold = 0.01

omega = batch_gradient_descent(
    omega_init, train_features_array, train_labels_array, learning_rate, threshold
)
print(omega)
