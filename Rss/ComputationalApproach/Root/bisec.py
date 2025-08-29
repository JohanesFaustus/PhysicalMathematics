def bisection(f, a, b, tol=1e-7, max_iter=1000):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    for _ in range(max_iter):
        c = (a + b) / 2.0
        if abs(f(c)) < tol or (b - a) / 2.0 < tol:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    raise RuntimeError("Maximum iterations exceeded.")

# Example usage:
# if __name__ == "__main__":
#     f = lambda x: x**2- x -6
#     root = bisection(f, -3, -1)
#     print("Root:", root)
f= lambda x: x**2-x-6
x1=2
x2=4
print(bisection(f,x1,x2))