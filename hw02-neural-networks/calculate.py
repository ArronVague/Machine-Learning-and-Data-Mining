import numpy as np

omega_1 = np.array([[0.1, 0.8], [0.4, 0.6], [0.3, 0.5]])
a_0 = np.array([0.455, 0.505])
b_1 = np.array([0.0, 0.0, 0.0])

z_1 = omega_1.dot(a_0) + b_1

print("z_1 = ", z_1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


a_1 = sigmoid(z_1)

print("a_1 = ", a_1)

omega_2 = np.array([[0.9, 0.2, 0.7], [0.6, 0.3, 0.7], [0.1, 0.8, 0.5]])
b_2 = np.array([0.0, 0.0, 0.0])

z_2 = omega_2.dot(a_1) + b_2

print("z_2 = ", z_2)

a_2 = sigmoid(z_2)

print("a_2 = ", a_2)

omega_3 = np.array([[0.4, 0.3, 0.5], [0.6, 0.2, 0.8]])
b_3 = np.array([0.0, 0.0])

z_3 = omega_3.dot(a_2) + b_3

print("z_3 = ", z_3)

a_3 = sigmoid(z_3)

print("a_3 = ", a_3)

y = np.array([(0.15 + 0.7) / 2, 0.7])

l_a_3 = a_3 - y
print("l_a_3 = ", l_a_3)


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


sd_prime_3 = sigmoid_derivative(z_3)
print("sd_prime_3 = ", sd_prime_3)

l_z_3 = l_a_3 * sd_prime_3
print("l_z_3 = ", l_z_3)

sd_prime_2 = sigmoid_derivative(z_2)
print("sd_prime_2 = ", sd_prime_2)

l_z_2 = omega_3.T.dot(l_z_3) * sd_prime_2
print("l_z_2 = ", l_z_2)

sd_prime_1 = sigmoid_derivative(z_1)
print("sd_prime_1 = ", sd_prime_1)

l_z_1 = omega_2.T.dot(l_z_2) * sd_prime_1
print("l_z_1 = ", l_z_1)

# 随机梯度下降
learning_rate = 0.8

omega_3 = omega_3 - learning_rate * l_z_3.reshape(2, 1).dot(a_2.reshape(1, 3))
print("omega_3 = ", omega_3)

b_3 = b_3 - learning_rate * l_z_3
print("b_3 = ", b_3)

omega_2 = omega_2 - learning_rate * l_z_2.reshape(3, 1).dot(a_1.reshape(1, 3))
print("omega_2 = ", omega_2)

b_2 = b_2 - learning_rate * l_z_2
print("b_2 = ", b_2)

omega_1 = omega_1 - learning_rate * l_z_1.reshape(3, 1).dot(a_0.reshape(1, 2))
print("omega_1 = ", omega_1)
