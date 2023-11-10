# 实验九：贝叶斯分类

## 代码

引入拉普拉斯平滑前后，代码变化的地方不大。因此，实现一个`Bayesian_classification`类，其中，构造函数`__init__(self, train, test)`，给定样本及标签计算后验概率的函数`pro(self, a, index)`以及预测正确率的函数`predict(self)`，这三个函数的实现方法相同。

唯二的两个区别为：

引入拉普拉斯平滑之前：cal_priori_probability(self)，cal_conditional_probability(self)

引入拉普拉斯平滑之后：cal_priori_probability_laplacian_smoothing(self)，cal_conditional_probability_laplacian_smoothing(self)

## 结果

```bash
Accuracy without Laplacian smoothing:  0.34
Accuracy with Laplacian smoothing:  0.74
```

## 总结

