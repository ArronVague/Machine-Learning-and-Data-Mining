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
