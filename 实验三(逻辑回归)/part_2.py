import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl

# 1)
train_frame = pd.read_csv("flower_train.csv")
test_frame = pd.read_csv("flower_test.csv")

print(train_frame)
print(train_frame.isnull().sum())

"""
train1_frame[['height', 'weight']] = train1_frame[['height', 'weight']].replace(0, np.NaN)
print(train1_frame)

height_column = train1_frame['height']
weight_column = train1_frame['weight']

mean_height_by_gender = train1_frame.groupby('sex')['height'].transform('mean')
mean_weight_by_gender = train1_frame.groupby('sex')['weight'].transform('mean')

train1_frame['height'].fillna(mean_height_by_gender, inplace=True)
train1_frame['weight'].fillna(mean_weight_by_gender, inplace=True)
"""
train_frame[["x1", "x2"]] = train_frame[["x1", "x2"]].replace(0, np.NaN)

x1_column = train_frame["x1"]
x2_column = train_frame["x2"]

mean_x1_by_type = train_frame.groupby("type")["x1"].transform("mean")
mean_x2_by_type = train_frame.groupby("type")["x2"].transform("mean")

train_frame["x1"].fillna(mean_x1_by_type, inplace=True)
train_frame["x2"].fillna(mean_x2_by_type, inplace=True)

print(train_frame)
print(train_frame.isnull().sum())

"""
train1_frame['sex'] = np.where(train1_frame['sex'] == 'Male', 0, 1)
print(train1_frame)
"""

train_frame["type"] = np.where(train_frame["type"] == "Iris-setosa", 0, 1)
test_frame["type"] = np.where(test_frame["type"] == "Iris-setosa", 0, 1)
print(train_frame)
print(test_frame)

"""
2)线性模型为 𝑦=𝜔𝑇𝑥
 ，在这里，我们将偏置量b当成模型参数 𝑤0
 ，并额外引入 𝑥0=1
 这一特征。请相应地往
 (测试集和训练集)
 添加 𝑥0=1
 这一特征。

tips:上一次实验中的矩阵求解析解的方法中，需要往特征中加入一列全1的特征量，此处类似。
"""

train_frame.insert(0, "x0", 1)
test_frame.insert(0, "x0", 1)
print(train_frame)
print(test_frame)
# print(train_frame['x1'])


def batch_gradient_descent(omega_init, sample, learning_rate, threshold):
    # print(omega_init)
    # print(sample)
    def omega_update(omega):
        while True:
            omega_new = omega.copy()
            # print(omega_new)
            total_list = [0] * len(sample)
            # print("hahaha")
            for i in range(len(sample)):
                wx = (
                    omega[0] * sample[i][0]
                    + omega[1] * sample[i][1]
                    + omega[2] * sample[i][2]
                )
                ewx = math.exp(wx)
                # print(ewx)
                total_list[i] = ewx / (1 + ewx) - sample[i][3]
            # print(total_list)
            for i in range(3):
                s = 0
                for j, total in enumerate(total_list):
                    s += total * sample[j][i]
                # print(s)
                # print(omega_new[i] - learning_rate * s / len(sample))
                omega_new[i] = omega_new[i] - learning_rate * s / len(sample)
            #     print(omega_new[i])
            # print(omega_new)
            if all(abs(x - y) < threshold for x, y in zip(omega_new, omega)):
                break
            omega = omega_new
        return omega

    omega = omega_update(omega_init)
    return omega


omega_init = [1, 1, 1]
learning_rate = 0.01
threshold = 0.001

train = np.array(train_frame)
omega = batch_gradient_descent(omega_init, train, learning_rate, threshold)
print(omega)

test = np.array(test_frame)
trained = [omega[0] + omega[1] * x[1] + omega[2] * x[2] for x in test]

# 取负的平均对数似然函数为损失函数，使用它计算loss值
loss = 0
for i in range(len(test)):
    loss += math.log(
        test[i][3] / (1 + math.exp(-trained[i]))
        + (1 - test[i][3]) * math.exp(-trained[i]) / (1 + math.exp(-trained[i]))
    )
loss = -loss / len(test)

print(loss)

# 5)使用训练后的逻辑回归模型对测试数据集'flower_test.csv'进行预测，输出可视化结果（比如用seaborn或者matplotlib等可视化库来画出测试数据的散点图以及训练好的模型函数图像)，要求如下:
# 1.将所得到的逻辑回归模型所得到的决策边界绘制出来
# 2.测试集的所有点在同一幅图中进行绘制
# 3.需要给不同类别的测试点不同颜色，方便通过颜色的区别直观看到预测正确和错误的样本

# 确定图画边界和大小
plt.figure(figsize=(10, 5))
x_min, x_max = 0, 10
y_min, y_max = 0, 5
# 使用numpy中的meshgrid生成网格矩阵，方便进行之后的描点
boundary_x, boundary_y = np.meshgrid(
    np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
)
grid = np.c_[boundary_x.ravel(), boundary_y.ravel()]
# 加入偏置(或w_0)对应的特征为1的一列
e = np.ones((len(grid), 1))
grid = np.c_[e, grid]
# 假定下列的模型参数
w = np.array([[omega[0]], [omega[1]], [omega[2]]])
# 计算出网格点中每个点对应的逻辑回归预测值
z = grid.dot(w)
for i in range(len(z)):
    z[i][0] = 1 / (1 + np.exp(-z[i][0]))
    if z[i][0] < 0.5:
        z[i][0] = 0
    else:
        z[i][0] = 1
# 转换shape以作出决策边界
z = z.reshape(boundary_x.shape)
plt.contourf(boundary_x, boundary_y, z, cmap=plt.cm.Spectral, zorder=1)

class_1 = test_frame[test_frame["type"] == 1]
class_0 = test_frame[test_frame["type"] == 0]
plt.scatter(class_1["x1"], class_1["x2"], c="blue")
plt.scatter(class_0["x1"], class_0["x2"], c="red")
plt.show()


"""
可使用plt.scatter来绘制出测试集的每个样本点，并设置指定颜色来区分预测正确和错误的样本
plt.scatter(x,y,c="color")，x、y为坐标值，c为指定颜色
"""
