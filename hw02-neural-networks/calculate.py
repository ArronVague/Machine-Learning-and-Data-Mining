import numpy as np

omega_1 = np.array([[0.1, 0.8], [0.4, 0.6], [0.3, 0.5]])
a_0 = np.array([0.455, 0.505])

b_0 = np.array([0.0, 0.0, 0.0])

z_1 = omega_1.dot(a_0) + b_0

print("z_1 = ", z_1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


a_1 = sigmoid(z_1)

print("a_1 = ", a_1)
