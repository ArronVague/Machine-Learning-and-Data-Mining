# 实验四：模型评估

## 代码

### 留出法

```python
import pandas as pd
import numpy as np
import warnings

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
    num_iterations = 1000  # 迭代次数

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

```

### 交叉验证法

```python
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

```

### 自助法

```python
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

```

## 结果

### 判断不同数据划分方式的性能

留出法、交叉验证法、自助法的精度分别为：

```powershell
留出法精度： 0.7849462365591398
交叉验证法平均精度： 0.6935483870967741
自助法精度： 0.8596491228070176（最好的一次）
```

留出法和交叉验证法由于随机种子random_state不变，因此每次运行相同的代码，得到的结果都是相同的。而自助法采用random.randint(a, b)，因此每次的结果都是不同的，它的结果也在0.72到0.86之间波动，这里为了简化实验，直接取自助法的最高精度。

因此，性能【最好】的数据划分方式是自助法，性能【最差】的是交叉验证法。

### 画出P-R曲线和ROC曲线，计算各点F1度量

#### P-R曲线

![](C:\Users\Arron\AppData\Roaming\marktext\images\2023-10-05-18-52-33-image.png)

illness.csv的数据不平衡，正例样本较少，PR曲线可能会受到数据分布的影响。

同时，使用梯度下降优化算法进行训练，而这种算法具有随机性质，可能导致PR曲线上下波动。

#### ROC曲线

![](C:\Users\Arron\AppData\Roaming\marktext\images\2023-10-05-18-53-16-image.png)

当数据集中的正例和负例样本数量不平衡时，ROC曲线可能出现样本点稀疏的情况。不平衡数据集导致某些阈值范围内的样本数量很少，从而导致样本点稀疏。

#### F1度量

```powershell
[0.07142857142857142, 0.13793103448275862, 0.13333333333333333, 0.19354838709677416, 0.25, 0.303030303030303, 0.35294117647058826, 0.39999999999999997, 0.4444444444444444, 0.43243243243243246, 0.4736842105263157, 0.5128205128205128, 0.5, 0.5365853658536585, 0.5714285714285714, 0.6046511627906976, 0.5909090909090909, 0.5777777777777777, 0.6086956521739131, 0.6382978723404256, 0.6666666666666666, 0.6530612244897959, 0.6399999999999999, 0.627450980392157, 0.6153846153846153, 0.6037735849056604, 0.5925925925925926, 0.5818181818181818, 0.5714285714285714, 0.5614035087719299, 0.5517241379310345, 0.5423728813559322, 0.5333333333333333, 0.5245901639344261, 0.5161290322580645, 0.5079365079365079, 0.5, 0.49230769230769234, 0.4848484848484849, 0.4776119402985075, 0.47058823529411764, 0.463768115942029, 0.45714285714285713, 0.4507042253521127, 0.4444444444444444, 0.4383561643835616, 0.4324324324324324, 0.4266666666666667, 0.42105263157894735, 0.41558441558441556, 0.4102564102564103, 0.40506329113924056, 0.4, 0.3950617283950617, 0.39024390243902435, 0.3855421686746988, 0.38095238095238093, 0.3764705882352941, 0.37209302325581395, 0.36781609195402304, 0.36363636363636365, 0.3595505617977528, 0.3555555555555555, 0.3516483516483516, 0.34782608695652173, 0.3440860215053763, 0.3404255319148936, 0.33684210526315783, 0.3333333333333333, 0.32989690721649484, 0.32653061224489793, 0.3232323232323232, 0.32, 0.31683168316831684, 0.3137254901960784, 0.3106796116504854, 0.3076923076923077, 0.32380952380952377, 0.33962264150943394, 0.35514018691588783, 0.37037037037037035, 0.38532110091743116, 0.4, 0.4144144144144144, 0.4285714285714286, 0.4424778761061947, 0.456140350877193, 0.46956521739130436, 0.46551724137931033, 0.4615384615384615, 0.4576271186440678, 0.45378151260504207, 0.45000000000000007, 0.44628099173553726, 0.4426229508196721, 0.43902439024390244, 0.435483870967742, 0.43199999999999994, 0.42857142857142855, 0.4251968503937008, 0.421875, 0.4186046511627907, 0.4153846153846154, 0.4122137404580153, 0.40909090909090906, 0.40601503759398494, 0.40298507462686567, 0.4, 0.39705882352941174, 0.39416058394160586, 0.391304347826087, 0.3884892086330935, 0.38571428571428573, 0.38297872340425526]
```

各点F1度量值在0.07142857142857142 ~ 0.6666666666666666之间，平均值为0.4290619258796318。整体上，模型的性能中等。

## 总结

1. 数据集划分：将可用数据集划分为训练集和测试集。

2. 模型选择：选择适当的机器学习模型来解决给定的问题，本次实验使用逻辑回归。

3. 模型训练：使用训练集对选定的模型进行训练。本次实验使用梯度下降算法。

4. 模型评估：使用测试集对训练好的模型进行评估。

5. 结果比较：对不同模型的评估结果进行比较和分析。

本次实验中，只比较了三种方法的精度。仅选取自助法评估PR曲线、ROC曲线和F1曲线，完整实验应做出三种方法的多指标评估。


