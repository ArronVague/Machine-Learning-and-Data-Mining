# 实验九：贝叶斯分类

## 代码

引入拉普拉斯平滑前后，代码变化的地方不大。其中，基础参数初始化，给定样本及标签计算后验概率的函数`pro(a, index)`以及预测正确率的函数`predict()`，这几个部分完全相同。唯二不同的两个地方为：

1. 先验概率

   引入拉普拉斯平滑之前，先验概率的计算公式为：
   $$
   P(y)=\frac{|D_y|}{|D|}
   $$
   

   引入拉普拉斯平滑之后，先验概率的计算公式为：
   $$
   P(y)=\frac{|D_y|+1}{|D|+N}
   $$
   

   因此，仅需要在原本公式基础上，对代码稍作修改：

   ```python
       # 计算先验概率
       def cal_priori_probability(self):
           for k, v in self.label_count.items():
               self.priori_probability[k] = v / self.Dy
   
       # 计算先验概率，使用拉普拉斯平滑
       def cal_priori_probability_laplacian_smoothing(self):
           for k, v in self.label_count.items():
               self.priori_probability[k] = (v + 1) / (
                   self.Dy + self.label_count.keys().__len__()
               )
   ```

2. 条件概率

   条件概率的计算基本与先验概率相同，这里不做过多赘述。

因此，实现一个`Bayesian_classification`类。

### 完整代码

```python
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class Bayesian_classification:
    # 接受np.array类型的训练集和测试集，初始化各种参数
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.num_feature = train.shape[1] - 1
        self.feature_unique = [0] * self.num_feature
        for i in range(self.num_feature):
            self.feature_unique[i] = set(train[:, i])
        self.label_count = Counter(train[:, -1])
        self.Dy = len(train)
        self.D = {}
        for k in self.label_count.keys():
            self.D[k] = train[train[:, -1] == k]
        self.priori_probability = {}
        self.conditional_probability = {}

    # 计算先验概率
    def cal_priori_probability(self):
        for k, v in self.label_count.items():
            self.priori_probability[k] = v / self.Dy

    # 计算先验概率，使用拉普拉斯平滑
    def cal_priori_probability_laplacian_smoothing(self):
        for k, v in self.label_count.items():
            self.priori_probability[k] = (v + 1) / (
                self.Dy + self.label_count.keys().__len__()
            )

    # 计算条件概率
    def cal_conditional_probability(self):
        for i in range(self.num_feature):
            for feature in self.feature_unique[i]:
                for k in self.label_count.keys():
                    Dxy = self.D[k][self.D[k][:, i] == feature]
                    self.conditional_probability[(i, feature, k)] = len(Dxy) / len(
                        self.D[k]
                    )

    # 计算条件概率，使用拉普拉斯平滑
    def cal_conditional_probability_laplacian_smoothing(self):
        for i in range(self.num_feature):
            for feature in self.feature_unique[i]:
                for k in self.label_count.keys():
                    Dxy = self.D[k][self.D[k][:, i] == feature]
                    self.conditional_probability[(i, feature, k)] = (len(Dxy) + 1) / (
                        len(self.D[k]) + self.feature_unique[i].__len__()
                    )

    # 给定样本以及标签，计算其后验概率
    def pro(self, a, index):
        res = self.priori_probability[index]
        for i, x in enumerate(a):
            if (i, x, index) not in self.conditional_probability.keys():
                return 0
            res *= self.conditional_probability[(i, x, index)]
        return res

    # 预测测试集，返回准确率
    def predict(self):
        accuracy = 0
        for a in self.test:
            rec = 0
            predict = ""
            for label in self.label_count.keys():
                p = self.pro(a[:-1], label)
                if p >= rec:
                    rec = p
                    predict = label
            accuracy += 1 if predict == a[-1] else 0
        accuracy = accuracy / len(self.test)
        return accuracy


train_df = pd.read_csv("train_mushroom.csv")
test_df = pd.read_csv("test_mushroom.csv")

train = np.array(train_df)
test = np.array(test_df)

bc = Bayesian_classification(train, test)
bc.cal_priori_probability()
bc.cal_conditional_probability()
print("Accuracy without Laplacian smoothing: ", bc.predict())

bc.cal_priori_probability_laplacian_smoothing()
bc.cal_conditional_probability_laplacian_smoothing()
print("Accuracy with Laplacian smoothing: ", bc.predict())

```

## 结果

```bash
Accuracy without Laplacian smoothing:  0.34
Accuracy with Laplacian smoothing:  0.74
```

实验过程中发现，使得条件概率为0的情况经常出现，导致对后验概率的计算产生了极大的影响，正确率只有34%。引入了拉普拉斯平滑后，正确率翻了一倍不止。

## 总结

做实验的过程中，由于代码实现的原因，条件概率是使用哈希表存储的，会出现某个条件概率根本不存在的情况，此时用测试集测试，就会出现bug。因此，需要在计算后验概率时，先检验某个条件概率是否存在，如果存在，代码继续执行；如果不存在，直接返回0，对应条件概率为0的情况。

```python
    # 给定样本以及标签，计算其后验概率
    def pro(self, a, index):
        res = self.priori_probability[index]
        for i, x in enumerate(a):
            # 缺少了这一句判断的代码
            if (i, x, index) not in self.conditional_probability.keys():
                return 0
            res *= self.conditional_probability[(i, x, index)]
        return res
```

