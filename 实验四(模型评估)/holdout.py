import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import warnings
import random
from sklearn.metrics import precision_recall_curve

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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(X_train, y_train, X_test):
    # 初始化参数
    num_features = X_train.shape[1]
    theta = np.zeros(num_features)

    # 使用梯度下降法训练模型
    alpha = 0.01  # 学习率
    num_iterations = 1000000  # 迭代次数

    for _ in range(num_iterations):
        z = np.dot(X_train, theta)
        h = sigmoid(z)
        gradient = np.dot(X_train.T, (h - y_train)) / len(X_train)
        theta -= alpha * gradient

    # 在测试集上进行预测
    y_pred = sigmoid(np.dot(X_test, theta))
    y_pred = np.where(y_pred >= 0.5, 1, 0)

    return y_pred


def accuracy(y_true, y_pred):
    # 计算精度
    return np.sum(y_true == y_pred) / len(y_true)


y_pred = logistic_regression(train_features, train_labels, test_features)
acc = accuracy(test_labels, y_pred)
print("留出法精度：", acc)


def precision_recall_curve(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)[::-1]  # 按正例可能性降序排列的索引
    sorted_scores = y_scores[sorted_indices]  # 排序后的正例可能性
    sorted_labels = y_true[sorted_indices]  # 排序后的真实标签

    precision_values = []  # 存储精确率值
    recall_values = []  # 存储召回率值
    true_positives = 0  # 真正例数
    false_positives = 0  # 假正例数
    positive_count = np.sum(sorted_labels)  # 正例总数

    for label in sorted_labels:
        if label == 1:
            true_positives += 1
        else:
            false_positives += 1

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / positive_count

        precision_values.append(precision)
        recall_values.append(recall)

    return precision_values, recall_values


y_test = np.array(test_labels)

precision_values, recall_values = precision_recall_curve(y_test, y_pred)

# 绘制查准率-查全率曲线
plt.plot(recall_values, precision_values)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True)
plt.show()
