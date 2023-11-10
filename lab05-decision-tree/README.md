# 实验五：决策树

## 任务一

### 代码

```py
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

'''
使用pandas库将训练数据集'train_titanic.csv'载入到Dataframe对象中
'''
df = pd.read_csv("train_titanic.csv")

'''
给定任何标签数组计算其信息熵
输入：标签数组
输出：该数组对应的信息熵
'''
def entropy(label):
    # 使用numpy中的unique实现计数
    unique_labels, label_counts = np.unique(label, return_counts=True)
    total_samples = len(label)
    ent = 0

    for count in label_counts:
        probability = count / total_samples
        # 计算信息熵
        ent -= probability * np.log2(probability)

    return ent

'''
函数功能为将所给的数据集按照指定维度的特征进行划分为若干个不同的数据集
【输入】：特征集合，标签集合，指定维度
【输出】：划分后所得到的子树属性集合，子树标记集合
'''
def split(feature, label, d):
    # 使用numpy中的unique实现非重复值的提取
    unique_values = np.unique(feature[:, d])
    split_feature = []
    split_label = []

    for value in unique_values:
        indices = np.where(feature[:, d] == value)
        split_feature.append(feature[indices])
        split_label.append(label[indices])

    return split_feature, split_label

'''
函数功能为进行【一次】决策树的结点划分，遍历找出该特征集合中信息增益(使用ID3算法中的公式计算)【最大】的特征
输入：特征集合，标签集合
输出：该次划分的最佳信息增益值，最佳划分维度
'''
def one_split_ID3(x_data, y_label):
    num_samples = len(x_data)
    num_features = x_data.shape[1]
    base_entropy = entropy(y_label)
    best_entropy = 0.0
    best_dimension = None

    for feature_dim in range(num_features):
        feature_values = x_data[:, feature_dim]
        split_feature, split_label = split(x_data, y_label, feature_dim)
        new_entropy = 0.0

        for i in range(len(split_feature)):
            sub_feature = split_feature[i]
            sub_label = split_label[i]
            p = len(sub_feature) / num_samples
            new_entropy += p * entropy(sub_label)

        information_gain = base_entropy - new_entropy

        # 记录最佳的信息增益值和对应的特征的维数
        if information_gain > best_entropy:
            best_entropy = information_gain
            best_dimension = feature_dim

    return best_entropy, best_dimension

'''
函数功能为进行【一次】决策树的结点划分，遍历找出该特征集合中信息增益率(使用C4.5算法中的公式计算)【最大】的特征
输入：特征集合，标签集合
输出：最佳划分的信息增益率值，对应的划分维度
'''
def one_split_C4_5(x_data, y_label):
    num_features = x_data.shape[1]
    best_entropy = 0.0
    best_dimension = None

    for dimension in range(num_features):
        feature_values = x_data[:, dimension]
        split_feature, split_label = split(x_data, y_label, dimension)
        new_entropy = 0.0
        intrinsic_value = entropy(feature_values)

        for i in range(len(split_feature)):
            sub_features = split_feature[i]
            sub_labels = split_label[i]
            p = len(sub_features) / len(x_data)
            new_entropy += p * entropy(sub_labels)

        information_gain = entropy(y_label) - new_entropy
        gain_ratio = information_gain / intrinsic_value

        # 记录最佳的信息增益率和对应的特征维数
        if gain_ratio > best_entropy:
            best_entropy = gain_ratio
            best_dimension = dimension

    return best_entropy, best_dimension

'''
进行【一次】决策树的结点划分，遍历找出该特征集合中基尼系数(使用CART算法中的公式计算)最小的特征以及最佳的划分值
输入：特征集合，标签集合
输出：最佳的基尼系数，对应的划分维度，最佳划分值
'''
def one_split_CART(x_data, y_label):
    def gini_index(label):
        unique_labels, label_counts = np.unique(label, return_counts=True)
        total_samples = len(label)
        gini = 1

        for count in label_counts:
            probability = count / total_samples
            gini -= probability**2

        return gini

    def split_by_value(feature, label, value):
        indices = np.where(feature == value)
        split_label = label[indices]

        return split_label

    num_features = x_data.shape[1]
    best_entropy = float("inf")
    best_dimension = None
    best_value = None

    for dimension in range(num_features):
        feature_values = x_data[:, dimension]
        unique_values = np.unique(feature_values)

        for value in unique_values:
            split_label = split_by_value(feature_values, y_label, value)
            gini_index_left = gini_index(split_label)
            gini_index_right = gini_index(y_label) - gini_index_left
            total_samples = len(y_label)
            gini_index_dimension = (
                len(split_label) / total_samples
            ) * gini_index_left + (
                len(y_label) - len(split_label)
            ) / total_samples * gini_index_right

            # 记录最小的基尼系数、对应的特征维数和非重复值（分类值）
            if gini_index_dimension < best_entropy:
                best_entropy = gini_index_dimension
                best_dimension = dimension
                best_value = value

    return best_entropy, best_dimension, best_value

# 提取特征列
features = ["Sex", "sibsp", "Parch", "Pclass"]
x_data = df[features].values

# 提取标签列
y_label = df["Survived"].values


'''
应用之前你在第4、5、6个部分编写的三个函数，在训练数据集'train_titanic.csv'上依次使用这些函数进行【一次】结点划分，并输出对应的最佳特征维数以及相应的信息增益值/信息增益率/(基尼系数和分类值)
'''
ID3_best_entropy, ID3_best_dimension = one_split_ID3(x_data, y_label)
print("one_split_ID3:")
print("最佳特征维数:", ID3_best_dimension)
print("最佳信息增益值:", ID3_best_entropy)

C4_5_best_entropy, C4_5_best_dimension = one_split_C4_5(x_data, y_label)
print("one_split_C4_5:")
print("最佳特征维数:", C4_5_best_dimension)
print("最佳信息增益率:", C4_5_best_entropy)

CART_best_entropy, CART_best_dimension, CART_best_value = one_split_CART(
    x_data, y_label
)
print("one_split_CART:")
print("最佳特征维数:", CART_best_dimension)
print("最佳基尼系数:", CART_best_entropy)
print("最佳分类值:", CART_best_value)
```

### 结果

```powershell
one_split_ID3:
最佳特征维数: 0
最佳信息增益值: 0.10750711887455178
one_split_C4_5:
最佳特征维数: 0
最佳信息增益率: 0.11339165967945304
one_split_CART:
最佳特征维数: 2
最佳基尼系数: -0.13561570914735693
最佳分类值: 3
```

最佳基尼系数竟然出现了负值。回去翻看`one_split_CART`方法，发现此处出现错误

![](C:\Users\Arron\AppData\Roaming\marktext\images\2023-10-14-19-10-07-image.png)

在计算 `gini_index_right` 时，我们应该使用剩余样本集合 `y_label` 减去左子集 `split_label` 后的样本集合的基尼指数，而不是直接减去 `gini_index_left`。

### 修正后的结果

```powershell
one_split_ID3:
最佳特征维数: 0
最佳信息增益值: 0.10750711887455178
one_split_C4_5:
最佳特征维数: 0
最佳信息增益率: 0.11339165967945304
one_split_CART:
最佳特征维数: 1
最佳基尼系数: 0.0
最佳分类值: 5
```

## 任务二

### 代码

```py
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def entropy(label):
    unique_labels, label_counts = np.unique(label, return_counts=True)
    total_samples = len(label)
    ent = 0

    for count in label_counts:
        probability = count / total_samples
        ent -= probability * np.log2(probability)

    return ent

'''
【从剩余的特征集A中】寻找使得信息增益最大的特征
输入：数据集D、剩余的特征集A
输出：最佳划分的特征维数
'''
def best_split(D, A):
    def split_by_value(feature, label, value):
        indices = np.where(feature == value)
        split_label = label[indices]

        return split_label

    best_information_gain = 0
    best_dimension = None

    for dimension in A:
        feature_values = D[:, dimension]
        unique_values = np.unique(feature_values)
        information_gain = entropy(
            D[:, -1]
        )  # Initialize with the entropy of the whole dataset

        for value in unique_values:
            split_label = split_by_value(feature_values, D[:, -1], value)
            information_gain -= (len(split_label) / len(D)) * entropy(split_label)

        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_dimension = dimension

    return best_dimension


# 树结点类
class Node:
    def __init__(self, isLeaf=True, label=-1, feature_index=-1):
        self.isLeaf = isLeaf  # isLeaf表示该结点是否是叶结点
        self.label = label  # label表示该叶结点的label（当结点为叶结点时有用）
        self.feature_index = feature_index  # feature_index表示该分支结点的划分特征的序号（当结点为分支结点时有用）
        self.children = {}  # children表示该结点的所有孩子结点，dict类型，方便进行决策树的搜索

    def addNode(self, val, node):
        self.children[val] = node  # 为当前结点增加一个划分特征的值为val的孩子结点

'''
完成DTree类中的TreeGenerate、train函数以完成决策树的构建。并完成DTree类中的predict函数来用构建好的决策树来对测试数据集进行预测并输出预测准确率。
'''
# 决策树类
class DTree:
    def __init__(self):
        self.tree_root = None  # 决策树的根结点
        self.possible_value = {}  # 用于存储每个特征可能的取值

    """
    TreeGenerate函数用于递归构建决策树，伪代码参照课件中的“Algorithm 1 决策树学习基本算法”

    """

    def TreeGenerate(self, D, A):
        # 生成结点 node
        node = Node()

        # if D中样本全属于同一类别C then
        #     将node标记为C类叶结点并返回
        # end if
        if len(np.unique(D[:, -1])) == 1:
            node.isLeaf = True
            node.label = D[0, -1]
            return node

        # if A = Ø OR D中样本在A上取值相同 then
        #     将node标记叶结点，其类别标记为D中样本数最多的类并返回
        # end if
        tmp = np.array(list(A))
        if len(tmp) == 0 or len(np.unique(D[:, tmp])) == 1:
            node.isLeaf = True
            node.label = Counter(D[:, -1]).most_common(1)[0][0]
            return node

        # 从A中选择最优划分特征a_star
        # （选择信息增益最大的特征，用到上面实现的best_split函数）
        a_star = best_split(D, A)

        # for a_star 的每一个值a_star_v do
        #     为node 生成每一个分支；令D_v表示D中在a_star上取值为a_star_v的样本子集
        #     if D_v 为空 then
        #         将分支结点标记为叶结点，其类别标记为D中样本最多的类
        #     else
        #         以TreeGenerate(D_v,A-{a_star}) 为分支结点
        #     end if
        # end for
        for a_star_v in np.unique(D[:, a_star]):
            if len(D[D[:, a_star] == a_star_v]) == 0:
                node.addNode(
                    a_star_v, Node(True, Counter(D[:, -1]).most_common(1)[0][0])
                )
            else:
                node.addNode(
                    a_star_v,
                    self.TreeGenerate(
                        D[D[:, a_star] == a_star_v], A - {a_star}
                    ),  # 递归调用TreeGenerate函数
                )

        # def __init__(self, isLeaf=True, label=-1, feature_index=-1) node的构造函数中，isLeaf默认为True，feature_index默认为-1。而函数执行到这里时，node的值需要赋值为False，feature_index需要赋值为a_star
        node.isLeaf = False
        node.feature_index = a_star
        return node

    """
    train函数可以做一些数据预处理（比如Dataframe到numpy矩阵的转换，提取属性集等），并调用TreeGenerate函数来递归地生成决策树

    """

    def train(self, D):
        D = np.array(D)  # 将Dataframe对象转换为numpy矩阵（也可以不转，自行决定做法）
        A = set(range(D.shape[1] - 1))  # 特征集A

        # 记下每个特征可能的取值
        for every in A:
            self.possible_value[every] = np.unique(D[:, every])

        self.tree_root = self.TreeGenerate(D, A)  # 递归地生成决策树，并将决策树的根结点赋值给self.tree_root
        return

    """
    predict函数对测试集D进行预测， 并输出预测准确率（预测正确的个数 / 总数据数量）

    """

    def predict(self, D):
        D = np.array(D)  # 将Dataframe对象转换为numpy矩阵（也可以不转，自行决定做法）
        # 对于D中的每一行数据d，从当前结点x=self.tree_root开始，当当前结点x为分支结点时，
        # 则搜索x的划分特征为该行数据相应的特征值的孩子结点（即x=x.children[d[x.index]]），不断重复，
        # 直至搜索到叶结点，该叶结点的标签就是数据d的预测标签
        acc_num = 0
        for d in D:
            # print("d:", d)
            x = self.tree_root
            while not x.isLeaf:
                # print("x.feature_index:", x.feature_index)
                x = x.children[d[x.feature_index]]
            # print(x.label)
            if x.label == d[-1]:
                acc_num += 1
        acc = acc_num / len(D)
        return acc


df = pd.read_csv("train_titanic.csv")
test_df = pd.read_csv("test_titanic.csv")

# 构建决策树
tree = DTree()
tree.train(df)

# 利用构建好的决策树对测试数据集进行预测，输出预测准确率（预测正确的个数 / 总数据数量）
acc = tree.predict(test_df)
print("预测准确率：", acc)

#展示生成的决策树结构
def display_tree(node, indent=""):
    if node.isLeaf:
        print(indent + "Leaf Node: label =", node.label)
    else:
        print(indent + "Branch Node: feature_index =", node.feature_index)
        for value, child in node.children.items():
            print(indent + "|--> Child Value:", value)
            display_tree(child, indent + "   ")


display_tree(tree.tree_root)
```

### 结果

```powershell
预测准确率： 0.8415841584158416
Branch Node: feature_index = 0
|--> Child Value: 0
   Branch Node: feature_index = 2
   |--> Child Value: 0
      Branch Node: feature_index = 3
      |--> Child Value: 1
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 0
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 0
      |--> Child Value: 2
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 0
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 0
      |--> Child Value: 3
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 0
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 0
         |--> Child Value: 3
            Leaf Node: label = 0
   |--> Child Value: 1
      Branch Node: feature_index = 1
      |--> Child Value: 0
         Branch Node: feature_index = 3
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 0
         |--> Child Value: 3
            Leaf Node: label = 0
      |--> Child Value: 1
         Branch Node: feature_index = 3
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 0
         |--> Child Value: 3
            Leaf Node: label = 0
      |--> Child Value: 2
         Leaf Node: label = 1
      |--> Child Value: 3
         Leaf Node: label = 0
      |--> Child Value: 4
         Leaf Node: label = 0
   |--> Child Value: 2
      Branch Node: feature_index = 1
      |--> Child Value: 0
         Branch Node: feature_index = 3
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 1
         |--> Child Value: 3
            Leaf Node: label = 0
      |--> Child Value: 1
         Branch Node: feature_index = 3
         |--> Child Value: 1
            Leaf Node: label = 1
         |--> Child Value: 2
            Leaf Node: label = 0
         |--> Child Value: 3
            Leaf Node: label = 0
      |--> Child Value: 2
         Leaf Node: label = 0
      |--> Child Value: 3
         Leaf Node: label = 0
      |--> Child Value: 4
         Leaf Node: label = 0
      |--> Child Value: 5
         Leaf Node: label = 0
      |--> Child Value: 8
         Leaf Node: label = 0
   |--> Child Value: 3
      Leaf Node: label = 0
   |--> Child Value: 4
      Leaf Node: label = 0
   |--> Child Value: 5
      Leaf Node: label = 0
   |--> Child Value: 6
      Leaf Node: label = 0
   |--> Child Value: 9
      Leaf Node: label = 0
|--> Child Value: 1
   Branch Node: feature_index = 3
   |--> Child Value: 1
      Branch Node: feature_index = 2
      |--> Child Value: 0
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 1
         |--> Child Value: 1
            Leaf Node: label = 1
         |--> Child Value: 2
            Leaf Node: label = 0
      |--> Child Value: 1
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 1
         |--> Child Value: 1
            Leaf Node: label = 1
      |--> Child Value: 2
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 1
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 1
         |--> Child Value: 3
            Leaf Node: label = 1
      |--> Child Value: 3
         Leaf Node: label = 0
      |--> Child Value: 4
         Leaf Node: label = 0
   |--> Child Value: 2
      Branch Node: feature_index = 1
      |--> Child Value: 0
         Branch Node: feature_index = 2
         |--> Child Value: 0
            Leaf Node: label = 1
         |--> Child Value: 1
            Leaf Node: label = 1
         |--> Child Value: 2
            Leaf Node: label = 0
         |--> Child Value: 3
            Leaf Node: label = 0
      |--> Child Value: 1
         Branch Node: feature_index = 2
         |--> Child Value: 0
            Leaf Node: label = 1
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 0
         |--> Child Value: 3
            Leaf Node: label = 1
      |--> Child Value: 2
         Branch Node: feature_index = 2
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 3
            Leaf Node: label = 1
   |--> Child Value: 3
      Branch Node: feature_index = 2
      |--> Child Value: 0
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 0
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 0
         |--> Child Value: 3
            Leaf Node: label = 1
      |--> Child Value: 1
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 0
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 1
         |--> Child Value: 3
            Leaf Node: label = 0
      |--> Child Value: 2
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 0
         |--> Child Value: 1
            Leaf Node: label = 0
         |--> Child Value: 2
            Leaf Node: label = 0
         |--> Child Value: 4
            Leaf Node: label = 0
         |--> Child Value: 5
            Leaf Node: label = 0
         |--> Child Value: 8
            Leaf Node: label = 0
      |--> Child Value: 3
         Leaf Node: label = 1
      |--> Child Value: 4
         Leaf Node: label = 0
      |--> Child Value: 5
         Branch Node: feature_index = 1
         |--> Child Value: 0
            Leaf Node: label = 0
         |--> Child Value: 1
            Leaf Node: label = 1
      |--> Child Value: 9
         Leaf Node: label = 0
```

ID3算法构建的决策树，它的预测准确率达到了0.8415841584158416。

## 总结

本次决策树实验，仅仅使用了ID3算法生成了完整的决策树，有着高达0.8415841584158416的准确率。不知道C4.5算法和CART算法生成的决策树，准确率能到达多少。


