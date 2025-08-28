import numpy as np

x = 1.0        # example value, set your own
u0 = x
x2 = x * x
res = u0
error = 1000
k = 1
e = 10**(-3)   # use **, not ^

while True:
    u0 = -u0 * x2 / ((2*k) * (2*k+1))
    res = res + u0
    error = np.abs(u0)

    if error < e:   # stop when error is small enough
        break

    k = k + 1

print("Result:", res)
