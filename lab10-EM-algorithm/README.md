# 实验十：EM算法

## 学习EM算法

[[5分钟学算法] #06 EM算法 你到底是哪个班级的](https://www.bilibili.com/video/BV1RT411G7jJ/?spm_id_from=333.337.search-card.all.click&vd_source=212ff176b778171e26249f81cfb5bdbc)

## 代码

利用python传递列表时，传递的是对象的引用而不是对象的副本这一特性，编写更新参数的代码时，不需要将参数列表返回，原地更新即可。但是要注意更新参数时参数之间的依赖关系，不能立即修改参数值。

```python
import numpy as np
import pandas as pd
import math

# 将数据集'data.csv'载入并转换为numpy数组
D = pd.read_csv("data.csv")
D = np.array(D)

# parameter = [alpha1, alpha2, mu1, mu2, sigma1, sigma2]
# 初始化参数
parameter = [0.625, 0.375, 175, 165, 4, 6]


# 概率密度函数 f(x|theta)
def f(x, parameter, i):
    # print(x)
    mui = parameter[i + 1]
    sigmai = parameter[i + 3]
    return math.exp(-((x - mui) ** 2) / (2 * (sigmai**2))) / (
        math.sqrt(2 * math.pi) * sigmai
    )


# P(x, z|theta)
def P(x, parameter, z):
    alphai = parameter[z - 1]
    return alphai * f(x, parameter, z)


# y1,i = P(z=1|x,theta)
def Y(x, parameter, z):
    return P(x, parameter, z) / (P(x, parameter, 1) + P(x, parameter, 2))


# 计算对数似然函数
def Q(x, parameter):
    return Y(x, parameter, 1) * math.log(P(x, parameter, 1)) + Y(
        x, parameter, 2
    ) * math.log(P(x, parameter, 2))


# 更新alpha
def alpha_expection(D, parameter):
    numerator1 = 0
    numerator2 = 0
    for x in D:
        numerator1 += Y(x[0], parameter, 1)
        numerator2 += Y(x[0], parameter, 2)
    n = len(D)
    parameter[0] = numerator1 / n
    parameter[1] = numerator2 / n


# 更新mu
def mu_expection(D, parameter):
    numerator1 = 0
    numerator2 = 0
    denominator1 = 0
    denominator2 = 0

    for x in D:
        numerator1 += Y(x[0], parameter, 1) * x[0]
        numerator2 += Y(x[0], parameter, 2) * x[0]
        denominator1 += Y(x[0], parameter, 1)
        denominator2 += Y(x[0], parameter, 2)

    parameter[2] = numerator1 / denominator1
    parameter[3] = numerator2 / denominator2


# 更新sigma
def sigma_expection(D, parameter, mu_next_1, mu_next_2):
    numerator1 = 0
    numerator2 = 0
    denominator1 = 0
    denominator2 = 0

    for x in D:
        numerator1 += Y(x[0], parameter, 1) * ((x[0] - mu_next_1) ** 2)
        numerator2 += Y(x[0], parameter, 2) * ((x[0] - mu_next_2) ** 2)
        denominator1 += Y(x[0], parameter, 1)
        denominator2 += Y(x[0], parameter, 2)

    parameter[4] = math.sqrt(numerator1 / denominator1)
    parameter[5] = math.sqrt(numerator2 / denominator2)


# 利用前面编写的函数完成EM算法的迭代过程，直至达到收敛要求。
# 收敛要求：
# 每轮参数更新的差值小于阈值

threshold = 0.0001

while True:
    record = parameter.copy()
    alpha_expection(D, parameter)
    mu_expection(D, parameter)

    mu_next_1 = parameter[2]
    mu_next_2 = parameter[3]
    sigma_expection(D, parameter, mu_next_1, mu_next_2)
    
    print(parameter)

    if all([abs(record[i] - parameter[i]) < threshold for i in range(len(parameter))]):
        break

```

## 结果

将阈值`threshold`设置为0.0001。

`parameter`参数列表对应关系如下：

```python
# parameter = [alpha1, alpha2, mu1, mu2, sigma1, sigma2]
```

### 第一次

直接使用数据集的正确参数作为初始参数。

```python
parameter = [0.625, 0.375, 175, 165, 4, 6]
```

```bash
[0.6295334901625836, 0.37046650983741736, 175.38382142996596, 165.01701860807503, 4.021720745856256, 5.635707775914282]
迭代次数： 44
```

### 第二次

以第七次全国人口普查以及《2022年中国居民身高体重健康数据报告》的数据为参考。

```python
parameter = [0.5124, 0.4876, 174, 161.4, 10, 10]
```

```bash
[0.6295297272028086, 0.37047027279718997, 175.3838417723788, 165.01708852927865, 4.021711647854404, 5.6357372294036905]
迭代次数： 271
```

### 第三次

```python
parameter = [0.5, 0.5, 140, 240, 10, 10]
```

```bash
[0.6300741503175135, 0.36992584968248715, 175.38089473118671, 165.0069686531602, 4.023029710652039, 5.631474155401422]
迭代次数： 358
```

```python
parameter = [0.5, 0.5, 240, 140, 10, 10]
```

```bash
[0.36992584968248715, 0.6300741503175135, 165.0069686531602, 175.38089473118671, 5.631474155401422, 4.023029710652039]
迭代次数： 358
```

似乎alpha都设置为0.5时，初始值较小的mu迭代完后的结果反而会比初始值大的mu迭代后的结果更大？

### 第四次

第三次得出的规律不存在。反例如下：

```python
parameter = [0.5, 0.5, 10, 300, 10, 10]
```

```bash
[0.36992796080096035, 0.6300720391990384, 165.00700790877917, 175.38090617461083, 5.631490692387824, 4.023024592336301]
迭代次数： 329
```

## 总结

实验的最终结果都接近数据集的正确相关信息。似乎只要在认知范围内设置初始参数，最终就会收敛到几乎一样的结果。

总的来说，EM算法就是先假设一组参数。然后

1. 基于这组参数，计算每个样本的属于不同类别的概率。（E步）
2. 然后用估计的分类来更新参数。（M步）

不断迭代，直到收敛。