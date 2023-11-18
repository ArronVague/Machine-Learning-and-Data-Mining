from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

D = pd.read_csv("data.csv")
D = np.array(D)

# parameter = [alpha1, alpha2, mu1, mu2, sigma1, sigma2]
parameter = [0.625, 0.375, 175, 165, 4, 6]


def f(x, parameter, i):
    # print(x)
    mui = parameter[i + 1]
    sigmai = parameter[i + 3]
    return math.exp(-((x - mui) ** 2) / (2 * (sigmai**2))) / (
        math.sqrt(2 * math.pi) * sigmai
    )


def P(x, parameter, z):
    alphai = parameter[z - 1]
    return alphai * f(x, parameter, z)


def Y(x, parameter, z):
    return P(x, parameter, z) / (P(x, parameter, 1) + P(x, parameter, 2))


def Q(x, parameter):
    return Y(x, parameter, 1) * math.log(P(x, parameter, 1)) + Y(
        x, parameter, 2
    ) * math.log(P(x, parameter, 2))


def alpha_expection(D, parameter):
    numerator1 = 0
    numerator2 = 0
    for x in D:
        numerator1 += Y(x[0], parameter, 1)
        numerator2 += Y(x[0], parameter, 2)
    n = len(D)
    parameter[0] = numerator1 / n
    parameter[1] = numerator2 / n


def mu_expection(D, parameter):
    numerator1 = 0
    numerator2 = 0
    denominator1 = 0
    denominator2 = 0

    for x in D:
        numerator1 += Y(x[0], parameter, 1) * x[0]
        numerator2 += Y(x[0], parameter, 2) * x[0]
        denominator1 += Y(x[0], parameter, 1)
        denominator2 += Y(x[0], parameter, 2)

    parameter[2] = numerator1 / denominator1
    parameter[3] = numerator2 / denominator2


def sigma_expection(D, parameter, mu_next_1, mu_next_2):
    numerator1 = 0
    numerator2 = 0
    denominator1 = 0
    denominator2 = 0

    for x in D:
        numerator1 += Y(x[0], parameter, 1) * ((x[0] - mu_next_1) ** 2)
        numerator2 += Y(x[0], parameter, 2) * ((x[0] - mu_next_2) ** 2)
        denominator1 += Y(x[0], parameter, 1)
        denominator2 += Y(x[0], parameter, 2)

    parameter[4] = math.sqrt(numerator1 / denominator1)
    parameter[5] = math.sqrt(numerator2 / denominator2)


# 利用前面编写的函数完成EM算法的迭代过程，直至达到收敛要求。


while True:
    record = parameter.copy()
    alpha_expection(D, parameter)
    mu_expection(D, parameter)

    mu_next_1 = parameter[2]
    mu_next_2 = parameter[3]
    sigma_expection(D, parameter, mu_next_1, mu_next_2)
    print(parameter)
    if all([abs(record[i] - parameter[i]) < 0.0001 for i in range(len(parameter))]):
        break
