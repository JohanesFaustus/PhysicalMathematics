import numpy as np


# Algoritm
# Bisection
def bisec(f, x1, x2):
    loop = 0
    xt = (x1 + x2) / 2
    eps = 10 ** (-17)
    if f(x1) * f(x2) >= 0:
        raise ValueError("no root within this range")
    else:
        while np.abs(f(xt)) > eps and np.abs(x1 - x2) > eps:
            if f(x1) * f(xt) < 0:
                x2 = xt
            else:
                x1 = xt
            xt = (x1 + x2) / 2
            loop += 1
            # print(xt, loop)
        return xt, loop


# Newton-Rhapson
def NR(f, df, a, eps=10 ** (-6)):
    loop = 0
    xr = 0
    while np.abs(f(a)) > eps:
        xr = a - f(a) / df(a)
        a = xr
        loop += 1
    return xr, loop


# Combination of bisec and NR
def BNR(f, dfdx, xi, xf, dx, eps=10 ** (-6)):
    roots = []
    loops = []
    a = xi
    while a < xf:
        b = a + dx
        if b > xf:
            break
        if f(a) * f(b) <= 0:
            xr = (a + b) / 2
            loop = 0
            while np.abs(f(xr)) > eps and loop < 100:
                xr = xr - f(xr) / dfdx(f, xr)
                loop += 1
            roots.append(float(xr))
            loops.append(loop)
        a = b
    if not roots:
        raise ValueError("No root found in interval")
    print("Akar fungsi adalah ", roots)
    Print("dengan jumlah looping ", loops)


# Various function
# Whatever trig function is this
def p(x):
    return np.sin(x**2) - np.sqrt(x) + x ** (1 / 3)


# Bessel function
def Jn(x, n=5, eps=10 ** (-6)):
    xo2 = x / 2.0
    x2 = xo2 * xo2
    a0 = 1.0

    for i in range(1, n + 1):
        a0 = a0 * xo2 / i

    ss = a0
    error = a0
    k = 1

    while error > eps:
        a0 = -a0 * x2 / ((n + k) * k)
        ss = ss + a0
        error = abs(a0)
        k += 1

    return ss


# Simple polynomial
def f(x):
    return (x - 3) * (x + 1)


# Derivative with cetral difference
def dfdx(f, x, h=10 ** (-5)):
    return (f(x + h) - f(x - h)) / (2 * h)


# Misc. usage
# print(bisec(f, 2.5, 4))
# (bisec(f, 2.5, 4))
