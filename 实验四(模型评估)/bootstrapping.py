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
print("训练集大小: ", len(train_data))
print("测试集大小: ", len(test_data))
