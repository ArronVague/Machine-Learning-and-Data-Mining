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

class_0_samples = df[df["class"] == 0]
class_1_samples = df[df["class"] == 1]

class_0_samples = class_0_samples.sample(frac=1, random_state=42)
class_1_samples = class_1_samples.sample(frac=1, random_state=42)

k = 5

train_data = []
test_data = []

for i in range(k):
    train_fold = pd.DataFrame()
    test_fold = pd.DataFrame()

    class_0_fold_size = len(class_0_samples) // k
    class_1_fold_size = len(class_1_samples) // k

    test_fold = test_fold.append(
        class_0_samples[i * class_0_fold_size : (i + 1) * class_0_fold_size]
    )
    test_fold = test_fold.append(
        class_0_samples[i * class_1_fold_size : (i + 1) * class_1_fold_size]
    )

    train_fold = df.drop(test_fold.index)

    train_fold = train_fold.sample(frac=1, random_state=42)
    test_fold = test_fold.sample(frac=1, random_state=42)

    train_data.append(train_fold)
    test_data.append(test_fold)

# for i in range(k):
#     print("第", i + 1, "折的测试集: ", (test_data[i].index))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义逻辑回归模型训练函数
def logistic_regression_train(X, y, learning_rate=0.01, num_iterations=10000):
    num_samples, num_features = X.shape
    theta = np.zeros(num_features)

    for _ in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)

        gradient = np.dot(X.T, (h - y)) / num_samples
        theta -= learning_rate * gradient

    return theta


# 定义逻辑回归模型预测函数
def logistic_regression_predict(X, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)

    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred


k = 5  # 折数

avg_accuracy = 0  # 平均精度

for i in range(k):
    # 获取第k折的训练集和测试集
    train_fold = train_data[i]
    test_fold = test_data[i]

    # 提取训练集的特征和标签
    X_train = train_fold.iloc[:, :-1].values
    y_train = train_fold.iloc[:, -1].values

    # 提取测试集的特征和标签
    X_test = test_fold.iloc[:, :-1].values
    y_test = test_fold.iloc[:, -1].values

    # 训练逻辑回归模型
    theta = logistic_regression_train(X_train, y_train)

    # 在测试集上进行预测
    y_pred = logistic_regression_predict(X_test, theta)

    # 计算并打印精度
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    avg_accuracy += accuracy
    print("第", i + 1, "折的精度:", accuracy)
avg_accuracy /= k
print("交叉验证法平均精度：", avg_accuracy)
