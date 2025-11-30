import numpy as np
from numpy.polynomial.laguerre import Laguerre


def rect(x, y):
    luas = 0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        luas += y[i] * dx
    return luas


def trapez(x, y):
    luas = 0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        luas += 0.5 * (y[i] + y[i + 1]) * dx
    return luas


def simpson(x, y):
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError(
            f"Number of subintervals must be even (len(x)-1 even), ), n = {n}"
        )
    h = (x[-1] - x[0]) / n
    luas = y[0] + y[-1]
    for i in range(1, n):
        if i % 2 == 0:
            luas += 2 * y[i]
        else:
            luas += 4 * y[i]
    return luas * h / 3


def GaussLaguerre(n, g):
    Ln = Laguerre.basis(n)
    Lnp1 = Laguerre.basis(n + 1)
    xi = Ln.roots()
    wi = xi / ((n + 1) * Lnp1(xi)) ** 2
    return np.sum(wi * np.exp(xi) * g(xi))


def func(x):
    gamma = 0.01
    return np.exp(-gamma * x**2)
    