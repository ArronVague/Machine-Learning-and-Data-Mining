import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import warnings
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score

warnings.filterwarnings("ignore")

df = pd.read_csv("illness.csv")
df["class"] = df["class"].map({"Abnormal": 0, "Normal": 1}).fillna(-1)

# print(df.isnull().sum())

features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

num_samples = df.shape[0]
train_data = []
test_data = []

for i in range(num_samples):
    index = random.randint(0, num_samples - 1)
    train_data.append(df.iloc[index])

# 创建测试集（排除训练集中已存在的样本）
for _, row in df.iterrows():
    is_in_train_data = False
    for train_row in train_data:
        if row.equals(train_row):
            is_in_train_data = True
            break
    if not is_in_train_data:
        test_data.append(row)

# 打印训练集和测试集大小
# print("训练集大小: ", len(train_data))
# print("测试集大小: ", len(test_data))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义逻辑回归模型训练函数
def logistic_regression_train(X, y, learning_rate=0.01, num_iterations=1000):
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


# 提取训练集的特征和标签
X_train = np.array([train_row[:-1] for train_row in train_data])
y_train = np.array([train_row[-1] for train_row in train_data])

# 提取测试集的特征和标签
X_test = np.array([test_row[:-1] for test_row in test_data])
y_test = np.array([test_row[-1] for test_row in test_data])

# 训练逻辑回归模型
theta = logistic_regression_train(X_train, y_train)

# 在测试集上进行预测
y_pred = logistic_regression_predict(X_test, theta)

# 计算并打印精度
accuracy = np.sum(y_test == y_pred) / len(y_test)
print("自助法精度：", accuracy)
# print(y_pred)
# print(y_test)


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

print(len(precision_values), len(recall_values))


def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


# 计算每个点的 F1 度量
f1_scores = [
    calculate_f1(precision, recall)
    for precision, recall in zip(precision_values, recall_values)
]

print(f1_scores)


def roc_curve(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)[::-1]  # 按正例可能性降序排列的索引
    sorted_scores = y_scores[sorted_indices]  # 排序后的正例可能性
    sorted_labels = y_true[sorted_indices]  # 排序后的真实标签

    tpr_values = []  # 存储真正例率值
    fpr_values = []  # 存储假正例率值
    true_positives = 0  # 真正例数
    false_positives = 0  # 假正例数
    positive_count = np.sum(sorted_labels)  # 正例总数
    negative_count = len(sorted_labels) - positive_count  # 负例总数

    for label in sorted_labels:
        if label == 1:
            true_positives += 1
        else:
            false_positives += 1

        tpr = true_positives / positive_count
        fpr = false_positives / negative_count

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return fpr_values, tpr_values


# 将样本按照正例可能性进行排序
sorted_indices = np.argsort(y_pred)[::-1]
sorted_labels = y_test[sorted_indices]

# 计算假正例率和真正例率
fpr_values, tpr_values = roc_curve(sorted_labels, y_pred)

# 绘制 ROC 曲线
plt.plot(fpr_values, tpr_values)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True)
plt.show()
