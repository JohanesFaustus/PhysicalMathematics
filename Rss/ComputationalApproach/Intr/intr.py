import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def LagInt(x_points, F_points):
    x = sp.Symbol("x")
    n = len(x_points)
    P = 0

    for i in range(n):
        L = 1
        for j in range(n):
            if i != j:
                L *= (x - x_points[j]) / (x_points[i] - x_points[j])
        P += F_points[i] * L

    return sp.expand(P)

x_i = [800, 864, 927, 991, 1055, 1119, 1183, 1247, 1311, 1375, 1439, 1502, 1566]
F_i = [0.87, 1.1, 1.5, 1.8, 2.2, 2.8, 3.7, 5, 6.8, 8.6, 10, 12.5, 15.5]


x=sp.Symbol("x")
x_graph = np.linspace(min(x_i), max(x_i), 100)
inter= sp.lambdify(x,LagInt(x_i,F_i),"numpy")
plt.plot(x_graph,inter(x_graph))
plt.scatter(x_i,F_i)
plt.show()


