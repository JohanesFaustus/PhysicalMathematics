import numpy as np


def gaussian_elimination(Omega, W):
    # Ensure W is a column vector (n,1)
    W = W.reshape(-1, 1).astype(float)

    # Form augmented matrix [Omega | W]
    n = W.shape[0]
    A = np.hstack([Omega.astype(float), W])

    # Forward elimination
    for k in range(n - 1):
        pivot = A[k, k]
        for i in range(k + 1, n):
            factor = A[i, k] / pivot
            A[i, k:] = A[i, k:] - factor * A[k, k:]

    # Back substitution
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        x[i, 0] = (A[i, -1] - np.dot(A[i, i + 1 : n], x[i + 1 :, 0])) / A[i, i]

    return x, A


def gauss_jordan(Omega, W):
    # Ensure W is a column vector (n,1)
    W = W.reshape(-1, 1).astype(float)

    # Form augmented matrix [Omega | W]
    n = W.shape[0]
    A = np.hstack([Omega.astype(float), W])

    # Forward elimination (Gaussian elimination step)
    for k in range(n - 1):
        pivot = A[k, k]
        for i in range(k + 1, n):
            factor = A[i, k] / pivot
            A[i, k:] = A[i, k:] - factor * A[k, k:]

    # Normalize pivots and eliminate above pivots (Jordan step)
    for k in range(n - 1, -1, -1):  # work backwards
        pivot = A[k, k]
        A[k, :] = A[k, :] / pivot  # normalize pivot to 1
        for i in range(k):
            factor = A[i, k]
            A[i, :] = A[i, :] - factor * A[k, :]  # eliminate above

    # The solution is now the last column
    sol = A[:, -1]
    return sol, A


def lu_decomp(Omega):
    n = Omega.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Compute U[i, j] for j >= i
        for j in range(i, n):
            U[i, j] = Omega[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        # Compute L[j, i] for j >= i
        for j in range(i, n):
            if i == j:
                L[i, i] = 1.0
            else:
                L[j, i] = (Omega[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[
                    i, i
                ]

    return L, U


def lu_solve(L, U, W):
    n = L.shape[0]
    y = np.zeros((n, 1))
    for i in range(n):
        y[i, 0] = W[i, 0] - sum(L[i, j] * y[j, 0] for j in range(i))
    x = np.zeros((n, 1))
    for i in reversed(range(n)):
        x[i, 0] = (y[i, 0] - sum(U[i, j] * x[j, 0] for j in range(i + 1, n))) / U[i, i]
    return x


# Example system
Omega = np.array([[1,1,0,0], [0,0,1,1], [1,0,1,0],[0,1,0,1],[1,0,0,1],[0,1,1,0]], dtype=float)
W = np.array([[12],[8], [11], [9],[7],[13]], dtype=float)

# L, U = lu_decomp(Omega)
# sol = lu_solve(L, U, W)
Omega_T_Omega = Omega.T @ Omega
Omega_T_W = Omega.T @ W

# Solve using your Gauss–Jordan function
sol, A = gauss_jordan(Omega_T_Omega, Omega_T_W)
print(sol)

