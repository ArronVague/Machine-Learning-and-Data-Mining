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

for i in range(k):
    print("第", i + 1, "折的测试集: ", (test_data[i].index))
