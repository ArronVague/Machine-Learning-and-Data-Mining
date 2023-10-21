# 实验六：随机森林

## 代码

本次实验的代码与实验五区别不大。

修改之处

### best_split(D, A)

先从特征集A中随机选取k个特征构成特征集A'。

```py
def best_split(D, A):
    d = len(A)
    k = max(np.log2(d), 1)
    # 随机选取k个特征
    A_prime = np.random.choice(list(A), int(k), replace=False)
    
    ...

    return best_dimension
```

### DTree.predict(self, D)

测试集中出现决策树无法划分的特征值时的情况时，不再对其进行预测，直接给定划分失败的样本标签-1。  

```py
    def predict(self, D):
        D = np.array(D)  # 将Dataframe对象转换为numpy矩阵（也可以不转，自行决定做法）
        # 对于D中的每一行数据d，从当前结点x=self.tree_root开始，当当前结点x为分支结点时，
        # 则搜索x的划分特征为该行数据相应的特征值的孩子结点（即x=x.children[d[x.index]]），不断重复，
        # 直至搜索到叶结点，该叶结点的标签就是数据d的预测标签
        label = []
        for d in D:
            x = self.tree_root
            succeed = True
            while not x.isLeaf:
                # print("x.feature_index:", x.feature_index)
                if d[x.feature_index] not in x.children:
                    succeed = False
                    break
                x = x.children[d[x.feature_index]]
            # print(x.label)
            if succeed:
                label.append(x.label)
            else:
                # 对其不再进行预测，直接给定划分失败的样本标签为-1
                label.append(-1)
        return label
```

### DTree.TreeGenerate(self, D, A)

修复了实验五潜在的bug：a_star可能为空。

当a_star为空时，直接将当前节点标记为叶子节点。

```py
    def TreeGenerate(self, D, A):
        ...
    
        if a_star is not None:
            ...
    
        else:
            node.isLeaf = True
            node.label = Counter(D[:, -1]).most_common(1)[0][0]
            return node


```

### Bootstrap采样

```py
train_frame = pd.read_csv('train_titanic.csv')

# Bootstrap 采样
n = 10
tree = [DTree()] * n
for i in range(n):
    D = np.array(train_frame)
    D = D[np.random.choice(D.shape[0], D.shape[0], replace=True)]
    D = pd.DataFrame(D)
    tree[i].train(D)
```

### 训练决策树并预测

```py
test_frame = pd.read_csv('test_titanic.csv')

# ----- Your code here -------
result = []
for t in tree:
    result.append(t.predict(test_frame))

# 相对多数投票
result = np.array(result)
res = []
for i in range(len(result[0])):
    res.append(Counter(result[:, i]).most_common(1)[0][0])

# print(res)

accuracy = np.sum(res == test_frame['Survived']) / len(test_frame)

print("Bootstrap采样的准确率为：", accuracy)
```

## 结果

测试10次，每次训练10棵预测数，并用相对多数投票法来对各棵决策树的预测结果相结合，准确率结果如下。平均0.8143564356435643的准确率，较之实验五单棵决策树接近0.85的准确率更低。

```powershell
第 1 次准确率： 0.7970297029702971
第 2 次准确率： 0.8118811881188119
第 3 次准确率： 0.8316831683168316
第 4 次准确率： 0.8316831683168316
第 5 次准确率： 0.7920792079207921
第 6 次准确率： 0.806930693069307
第 7 次准确率： 0.801980198019802
第 8 次准确率： 0.8465346534653465
第 9 次准确率： 0.8267326732673267
第 10 次准确率： 0.7970297029702971
平均准确率： 0.8143564356435643
```

## 总结

导致随机森林的准确率比单棵决策树低的原因可能是：

1. 集群可能起到了副作用。

2. 过拟合：随机森林有多棵决策树组成，每棵树都是独立训练的。每棵树都会对训练数据进行学习，可能会过度拟合训练数据，导致在测试数据上的准确率下降。相比之下，单棵决策树可能在训练数据上过拟合的风险较低。


