import numpy as np
import pandas as pd
from collections import Counter

def entropy(label):
    unique_labels, label_counts = np.unique(label, return_counts=True)
    total_samples = len(label)
    ent = 0

    for count in label_counts:
        probability = count / total_samples
        ent -= probability * np.log2(probability)

    return ent

'''
先从特征集 𝐴
 中先随机选取 𝑘
 个特征构成特征集 𝐴′
 ，再从 𝐴′
 中选取最佳划分的特征。 𝑘
 一般取 𝑚𝑎𝑥{𝑙𝑜𝑔2𝑑,1}
 ,  𝑑
 是 𝐴
 的元素的个数。你可使用特征的信息增益来决定最佳划分的特征。
【输入】：数据集D、特征集A
【输出】：随机特征集A'中最佳划分的特征维数
'''
def best_split(D, A):
    d = len(A)
    k = max(np.log2(d), 1)
    # 随机选取k个特征
    A_prime = np.random.choice(list(A), int(k), replace=False)

    def split_by_value(feature, label, value):
        indices = np.where(feature == value)
        split_label = label[indices]

        return split_label

    best_information_gain = 0
    best_dimension = None

    for dimension in A_prime:
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

# 记下所有属性可能的取值
train_frame = pd.read_csv('train_titanic.csv')
D = np.array(train_frame)
A = set(range(D.shape[1] - 1))
possible_value = {}
for every in A:
    possible_value[every] = np.unique(D[:, every])


# 树结点类
class Node:
    def __init__(self, isLeaf=True, label=-1, feature_index=-1):
        self.isLeaf = isLeaf  # isLeaf表示该结点是否是叶结点
        self.label = label  # label表示该叶结点的label（当结点为叶结点时有用）
        self.feature_index = feature_index  # feature_index表示该分支结点的划分特征的序号（当结点为分支结点时有用）
        self.children = {}  # children表示该结点的所有孩子结点，dict类型，方便进行决策树的搜索

    def addNode(self, val, node):
        self.children[val] = node  # 为当前结点增加一个划分特征的值为val的孩子结点


# 决策树类
class DTree:
    def __init__(self):
        self.tree_root = None  # 决策树的根结点
        self.possible_value = possible_value  # 用于存储每个特征可能的取值

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

        if a_star is not None:
            for a_star_v in np.unique(D[:, a_star]):
                D_v = D[D[:, a_star] == a_star_v]
                if len(D_v) == 0:
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
        else:
            node.isLeaf = True
            node.label = Counter(D[:, -1]).most_common(1)[0][0]
            return node

    """
    train函数可以做一些数据预处理（比如Dataframe到numpy矩阵的转换，提取属性集等），并调用TreeGenerate函数来递归地生成决策树
 
    """

    def train(self, D):
        D = np.array(D)  # 将Dataframe对象转换为numpy矩阵（也可以不转，自行决定做法）
        
        A = set(range(D.shape[1] - 1))  # 特征集A
        # print(A)

        self.tree_root = self.TreeGenerate(D, A)  # 递归地生成决策树，并将决策树的根结点赋值给self.tree_root
        return

    """
    predict(self, D)：对测试集D进行预测，要求返回数据集D的预测标签，即一个(|D|,1)矩阵（|D|行1列）。测试集中出现决策树无法划分的特征值时的情况时，对其不再进行预测，直接给定划分失败的样本标签(例如-1)。
    """

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
    
# Bootstrap采样
# 生成10个决策树
s = 0
cur = 0
while cur < 10:
    n = 1000
    tree = [DTree()] * n
    for i in range(n):
        D = np.array(train_frame)
        D = D[np.random.choice(D.shape[0], D.shape[0], replace=True)]
        D = pd.DataFrame(D)
        tree[i].train(D)

    # 测试
    test_frame = pd.read_csv('test_titanic.csv')
    result = []
    for t in tree:
        result.append(t.predict(test_frame))

    # 相对多数投票
    result = np.array(result)
    res = []
    for i in range(len(result[0])):
        res.append(Counter(result[:, i]).most_common(1)[0][0])


    accuracy = np.sum(res == test_frame['Survived']) / len(test_frame)
    s += accuracy
    print("第", cur + 1, "次准确率：", accuracy)
    cur += 1
s /= cur
print("平均准确率：", s)