from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

D = pd.read_csv("data.csv")

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
