import numpy as np
import matplotlib.pyplot as plt


def XTX_Construct(x_points, F_points, degree):
    x = np.array(x_points, dtype=float)
    F = np.array(F_points, dtype=float).reshape(-1, 1)
    N = len(x)

    # Construct design matrix manually
    X = np.zeros((N, degree + 1), dtype=float)
    for i in range(N):
        for j in range(degree + 1):
            X[i, j] = x[i] ** j  # x_i raised to the j-th power

    # Form normal equations
    XTX = X.T @ X
    XTF = X.T @ F

    return XTX, XTF
    # return X


def GaussElim(Omega, W):
    # Ensure W is a column vector (n,1)
    W = W.reshape(-1, 1).astype(float)
    # Form augmented matrix [Omega | W]
    n = W.shape[0]
    A = np.hstack([Omega.astype(float), W])

    # Forward elimination
    for k in range(n - 1):
        # Find pivot row
        pivot_row = np.argmax(np.abs(A[k:, k])) + k
        if A[pivot_row, k] == 0:
            raise ValueError("Matrix is singular.")
        # Swap rows if necessary
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]

        pivot = A[k, k]
        # Eliminate below pivot
        for i in range(k + 1, n):
            factor = A[i, k] / pivot
            A[i, k:] -= factor * A[k, k:]

    # Back substitution
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        x[i, 0] = (A[i, -1] - np.dot(A[i, i + 1 : n], x[i + 1 :, 0])) / A[i, i]

    return x, A


# Function to turn solution into polinomial
def func(sol, x):
    order = sol.shape[0]
    f = np.zeros_like(x, dtype=float)
    for i in range(order):
        f += sol[i,] * x**i
    return f


# Data (Stefan's experiment)
x_i = [800, 864, 927, 991, 1055, 1119, 1183, 1247, 1311, 1375, 1439, 1502, 1566]
F_i = [0.87, 1.1, 1.5, 1.8, 2.2, 2.8, 3.7, 5, 6.8, 8.6, 10, 12.5, 15.5]

# Init x array
x_graph = np.linspace(min(x_i), max(x_i), 100)

# matplotlib parameter to make the graph look good
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pgf.texsystem": "pdflatex",  # or xelatex/lualatex
        "pgf.rcfonts": False,  # Don't let Matplotlib redefine fonts
    }
)


# Plot the experiment data
plt.scatter(x_i, F_i, color="k", label="Experiment data")

# Construc curve for various order regression order
for j in range(4):
    XTX, XTF = XTX_Construct(x_i, F_i, j)
    sol, A = GaussElim(XTX, XTF)

    y_graph = func(sol, x_graph)
    plt.plot(x_graph, y_graph, label=rf"Regression of order $m={j}$")

# Show plot
plt.title(r"The $P$ vs $T$ Graph with various order of regression")
plt.xlabel(r"Temperature (K)")
plt.ylabel(r"Radiated Power (watt)")
plt.legend()
plt.show()


# Misc. stuff
# Func to turn list into data point in latex
# def DataPrint(x, F):
#     if len(x) != len(F):
#         print("Data point does not match")
#     else:
#         for i in range(len(x)):
#             # print(f"({x[i]}, {F[i]});", end=" ")
#             print(f"{F[i]}&", end=" ")
#         print()


# Function to turn array into matrix in latex
# def MatrixPrint(W):
#     m, n = W.shape
#     for i in range(m):
#         for j in range(n):
#             elem = W[i, j]
#             exp = int(np.floor(np.log10(abs(elem))))
#             coef = elem / 10**exp
#             if j != n:
#                 elem = f"{coef:.2f}\\times 10^{{{exp}}} &"
#             elem = f"{coef:.2f}\\times 10^{{{exp}}} "
#             print(elem, end=" ")
#         print("\\\\")


# XTX, XTF = XTX_Construct(x_i, F_i, 3)
# XTX = XTX_Construct(x_i, F_i, 3)
# sol, A = GaussElim(XTX, XTF)

# print(MatrixPrint(sol))
# print(XTX)
# print(MatrixPrint(XTF))
# DataPrint(x_i, F_i)
