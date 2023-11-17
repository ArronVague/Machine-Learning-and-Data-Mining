from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

D = pd.read_csv("data.csv")
D = np.array(D)

# parameter = [alpha1, alpha2, mu1, mu2, sigma1, sigma2]
parameter = [0] * 6


def f(x, parameter, i):
    mui = parameter[i + 1]
    sigmai = parameter[i + 3]
    return math.exp(-((x - mui) ** 2) / (2 * (sigmai**2))) / (
        math.sqrt(2 * math.pi) * sigmai
    )


def P(x, parameter, z):
    alphai = parameter[z - 1]
    if z == 1:
        return alphai * f(x, parameter, 1)
    elif z == 2:
        return alphai * f(x, parameter, 2)


def Y(x, parameter, z):
    if z == 1:
        return P(x, parameter, 1) / (P(x, parameter, 1) + P(x, parameter, 2))
    elif z == 2:
        return P(x, parameter, 2) / (P(x, parameter, 1) + P(x, parameter, 2))


def Q(x, parameter):
    return Y(x, parameter, 1) * math.log(P(x, parameter, 1)) + Y(
        x, parameter, 2
    ) * math.log(P(x, parameter, 2))


def alpha_expection(D, parameter):
    new_parameter = parameter.copy()
    new_parameter[0] = 0
    new_parameter[1] = 0
    for x in D:
        new_parameter[0] += Y(x, parameter, 1)
        new_parameter[1] += Y(x, parameter, 2)
    n = len(D)
    new_parameter[0] /= n
    new_parameter[1] /= n

    return new_parameter


def mu_expection(D, parameter):
    new_parameter = parameter.copy()
    new_parameter[0] = 0
    new_parameter[1] = 0
    base0 = 0
    base1 = 0
    for x in D:
        new_parameter[0] += Y(x, parameter, 1) * x
        new_parameter[1] += Y(x, parameter, 2) * x
        base0 += Y(x, parameter, 1)
        base1 += Y(x, parameter, 2)
    new_parameter[0] /= base0
    new_parameter[1] /= base1

    return new_parameter


def sigma_expection(D, parameter, mu_next_1, mu_next_2):
    new_parameter = parameter.copy()
    new_parameter[4] = 0
    new_parameter[5] = 0
    base4 = 0
    bese5 = 0
    for x in D:
        new_parameter[4] += Y(x, parameter, 1) * ((x - mu_next_1) ** 2)
        new_parameter[5] += Y(x, parameter, 2) * ((x - mu_next_2) ** 2)
        base4 += Y(x, parameter, 1)
        bese5 += Y(x, parameter, 2)
    new_parameter[4] /= base4
    new_parameter[5] /= bese5
    return new_parameter
