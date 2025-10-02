import numpy as np

def gaussian_elimination(Omega, W):
    # Ensure W is a column vector (n,1)
    W = W.reshape(-1, 1).astype(float)
    
    # Form augmented matrix [Omega | W]
    n = W.shape[0]
    A = np.hstack([Omega.astype(float), W])
    
    # Forward elimination
    for k in range(n-1):
        pivot = A[k, k]
        for i in range(k+1, n):
            factor = A[i, k] / pivot
            A[i, k:] = A[i, k:] - factor * A[k, k:]
    
    # Back substitution
    x = np.zeros((n,1))
    for i in range(n-1, -1, -1):
        x[i,0] = (A[i, -1] - np.dot(A[i, i+1:n], x[i+1:,0])) / A[i, i]
    
    return x, A

# Example system
Omega = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]], dtype=float)

W = np.array([[8],
              [-11],
              [-3]], dtype=float)

print(gaussian_elimination(Omega, W))

