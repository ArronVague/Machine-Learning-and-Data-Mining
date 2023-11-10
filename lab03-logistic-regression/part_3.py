import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


train_frame = pd.read_csv("train_titanic.csv")
test_frame = pd.read_csv("test_titanic.csv")

# 0 2 4没用
train_frame = train_frame.drop(columns=["Passengerid", "Fare", "sibsp"])
test_frame = test_frame.drop(columns=["Passengerid", "Fare", "sibsp"])
train_frame.insert(0, "x0", 1)
test_frame.insert(0, "x0", 1)


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
                    + omega[3] * sample[i][3]
                    + omega[4] * sample[i][4]
                )
                ewx = math.exp(wx)
                # print(ewx)
                total_list[i] = ewx / (1 + ewx) - sample[i][5]
            # print(total_list)
            for i in range(5):
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


omega_init = [8] * 5
learning_rate = 0.01
threshold = 0.0001
train = np.array(train_frame)
omega = batch_gradient_descent(omega_init, train, learning_rate, threshold)
print(omega)


test = np.array(test_frame)
trained = [
    omega[0] + omega[1] * x[1] + omega[2] * x[2] + omega[3] * x[3] + omega[4] * x[4]
    for x in test
]

# 取负的平均对数似然函数为损失函数，使用它计算loss值
loss = 0
for i in range(len(test)):
    loss += math.log(
        test[i][5] / (1 + math.exp(-trained[i]))
        + (1 - test[i][5]) * math.exp(-trained[i]) / (1 + math.exp(-trained[i]))
    )
loss = -loss / len(test)

print(loss)
