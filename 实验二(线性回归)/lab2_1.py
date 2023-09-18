import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_frame = pd.read_csv('train.csv')
test_frame = pd.read_csv('test.csv')

train = np.array(train_frame)
test = np.array(test_frame)

# 方法①
def func_1():
    x_average = np.mean(train[:, 0])
    omega_1 = np.sum((train[:, 0] - x_average) * train[:, 1]) / (np.sum(train[:, 0] ** 2) - np.sum(train[:, 0]) ** 2 / len(train))
    b_1 = np.mean(train[:, 1] - omega_1 * train[:, 0])
    return omega_1, b_1

# 方法②
# 将数据集分成k个mini_batch
def func_2(omega_init, b_init, k, learning_rate, threshold):
    def get_mini_batches(train, k):
        np.random.shuffle(train)
        mini_batches = [train[i:i + k] for i in range(0, len(train), k)]
        return mini_batches
    
    mini_batches = get_mini_batches(train, k)

    def gradient_descent(omega, b, mini_batches):
        # omega的更新公式
        def omega_gd(omega, b, mini_batch):
            total = 0
            for x, y in mini_batch:
                total += x * (omega * x + b - y)
            return omega - learning_rate * total / len(mini_batch)

        # b的更新公式
        def b_gd(omega, b, mini_batch):
            total = 0
            for x, y in mini_batch:
                total += omega * x + b - y
            return b - learning_rate * total / len(mini_batch)
        
        while True:
            random_index = np.random.randint(0, len(mini_batches))
            mini_batch = mini_batches[random_index]
            omega_new, b_new = omega_gd(omega, b, mini_batch), b_gd(omega, b, mini_batch)
            # 终止条件为参数更新的幅度小于阈值threshold
            if abs(omega_new - omega) < threshold and abs(b_new - b) < threshold:
                break
            omega, b = omega_new, b_new
        return omega, b
    return gradient_descent(omega_init, b_init, mini_batches)


test_x = test[:, 0]
test_y = test[:, 1]


omega_1, b_1 = func_1()
print('func_1: omega = %f, b = %f' % (omega_1, b_1))

#omega和b为任意初始值，k取10，学习率取0.01，阈值取0.0001
k = 10
learning_rate = 0.01
threshold = 0.0001

omega_2, b_2 = func_2(1, 1, k, learning_rate, threshold)
print('func_2: omega = %f, b = %f' % (omega_2, b_2))

train_y_1 = omega_1 * test_x + b_1
train_y_2 = omega_2 * test_x + b_2

plt.plot(test_x, train_y_1)
plt.plot(test_x, train_y_2)
plt.plot(test_x, test_y, '.')
plt.show()