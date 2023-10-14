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


"""
【从剩余的特征集A中】寻找使得信息增益最大的特征
输入：数据集D、剩余的特征集A
输出：最佳划分的特征维数
"""


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


"""
完成DTree类中的TreeGenerate、train函数以完成决策树的构建。并完成DTree类中的predict函数来用构建好的决策树来对测试数据集进行预测并输出预测准确率。
"""


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


# 展示生成的决策树结构
def display_tree(node, indent=""):
    if node.isLeaf:
        print(indent + "Leaf Node: label =", node.label)
    else:
        print(indent + "Branch Node: feature_index =", node.feature_index)
        for value, child in node.children.items():
            print(indent + "|--> Child Value:", value)
            display_tree(child, indent + "   ")


display_tree(tree.tree_root)
