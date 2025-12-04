# Example of numerical differential equation solution
import numpy as np 
import matplotlib.pyplot as plt

def linearconv(u, nt, CFL):
    nx = len(u)
    dx = 2 / (nx - 1)
    nt = 20  # nt is the number of timesteps we want to calculate
    c = 1
    sigma = 0.5
    dt = sigma * dx

    for n in range(nt):  # iterate through time
        un = u.copy()  ##copy the existing values of u into un
        for i in range(1, nx):
            u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1])


def linearconv2D(u, nt, CFL):
    nx = len(u)
    dx = 2 / (nx - 1)
    dy = dx
    c = 1
    sigma = CFL
    dt = sigma * dx

    for n in range(nt + 1):  ##loop across number of time steps
        un = u.copy()
        u[1:, 1:] = (
            un[1:, 1:]
            - (c * dt / dx * (un[1:, 1:] - un[1:, :-1]))
            - (c * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
        )
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    return u


def couplednonlinearconv2d(u, v, nt, CFL):
    nx = len(u)
    dx = 2 / (nx - 1)
    dy = dx
    c = 1
    sigma = CFL
    dt = sigma * dx

    for n in range(nt + 1):  ##loop across number of time steps
        un = u.copy()
        vn = v.copy()
        u[1:, 1:] = (
            un[1:, 1:]
            - (un[1:, 1:] * c * dt / dx * (un[1:, 1:] - un[1:, :-1]))
            - vn[1:, 1:] * c * dt / dy * (un[1:, 1:] - un[:-1, 1:])
        )
        v[1:, 1:] = (
            vn[1:, 1:]
            - (un[1:, 1:] * c * dt / dx * (vn[1:, 1:] - vn[1:, :-1]))
            - vn[1:, 1:] * c * dt / dy * (vn[1:, 1:] - vn[:-1, 1:])
        )

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    return u, v


def diffus(u, nt, CFL):
    nx = len(u)
    dx = 2 / (nx - 1)
    nu = 0.3
    sigma = CFL
    dt = sigma * dx**2 / nu

    for i in range(1, nt):
        un = u.copy()
        for j in range(1, nx - 1):
            u[j] = un[j] + nu * dt / dx**2 * (u[j + 1] - 2 * u[j] + u[j - 1])
    return u


def diffuse2d(u, nt, CFL):
    nx = len(u)
    ny = len(u[0])
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = sigma * dx
    nu = 0.05

    for n in range(nt + 1):
        un = u.copy()
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            + nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
            + nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
        )
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
    return u


def burger(u, nt, CFL):
    nx = len(u)
    dx = 2 * numpy.pi / (nx - 1)
    nu = 0.05
    dt = CFL * dx
    for n in range(nt):
        un = u.copy()
        for i in range(1, nx - 1):
            u[i] = (
                un[i]
                - un[i] * dt / dx * (un[i] - un[i - 1])
                + nu * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1])
            )
        u[-1] = u[0]

    return u


def burger2d(u, v, nt, CFL):
    nx = len(u)
    ny = nx
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    nu = 0.01
    dt = CFL * dx * dy / nu

    for n in range(nt + 1):  ##loop across number of time steps
        un = u.copy()
        vn = v.copy()

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
            - dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
            + nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
            + nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
            - dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
            + nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
            + nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
        )

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    return u, v


def laplace2d(p, y, dx, dy, l1norm_target):
    l1norm = 1

    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = (
            dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2])
            + dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])
        ) / (2 * (dx**2 + dy**2))

        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = 2  # p = y @ x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1
        l1norm = numpy.sum(numpy.abs(p[:]) - numpy.abs(pn[:])) / numpy.sum(
            numpy.abs(pn[:])
        )

    return p


# Surface plotting function
def surfaceplot(u, nx):
    x = numpy.linspace(0, 2, nx)
    y = numpy.linspace(0, 2, nx)
    X, Y = numpy.meshgrid(x, y)
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        X, Y, u, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=True
    )


# Bonus Initial condition


def square(nx):
    u = np.zeros(nx)
    start = nx // 8
    end = nx // 4

    u[start:end] = 1
    return u


def gaussian(nx, x0=0.5, sigma=0.1):
    x = np.linspace(0, 2, nx)
    u = np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    return u


# Differentiation method

def RK2Coup(f, t0, y0, h, t_end):
    t_rk2 = np.arange(t0, t_end + h, h)
    y_rk2 = np.zeros((len(t_rk2), len(y0)), dtype=float)
    y_rk2[0] = y0

    for i in range(len(t_rk2) - 1):
        t_i = t_rk2[i]
        y_i = y_rk2[i]

        k1 = f(t_i, y_i)
        k2 = f(t_i + h, y_i + h * k1)

        y_rk2[i + 1] = y_i + 0.5 * h * (k1 + k2)

    return t_rk2, y_rk2

# Appendix problem


def Q(p, q):
    return np.sqrt(2 / 5) * np.sin(p * q * np.pi / 5)


def omega(k):
    return 2 *np.sqrt(10) * np.sin(k * np.pi / 10)


def C(p, q, t):
    s = 0.0
    for k in range(1, 5):
        s += Q(p, k) * Q(q, k) * np.cos(omega(k) * t)
    return s


def S(p, q, t):
    s = 0.0
    for k in range(1, 5):
        ok = omega(k)
        s += Q(p, k) * Q(q, k) * np.sin(ok * t) / ok
    return s


def L(p, q, t):
    s = 0.0
    for k in range(1, 5):
        ok = omega(k)
        s += -Q(p, k) * Q(q, k) * ok * np.sin(ok * t)
    return s


def xi(i, t, x0, v0):
    s = 0.0
    for j in range(1, 5):
        s += C(i, j, t) * x0[j - 1] + S(i, j, t) * v0[j - 1]
    return s


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pgf.texsystem": "pdflatex",  
        "pgf.rcfonts": False, 
    }
)


t = np.arange(0,2*np.pi,0.001)
x0 = np.array(
    [[0.1],[0],[0],[-0.1]],
    dtype=float,
) 
v0 = np.array(
    [[0],[0],[0],[0]],
    dtype=float,
) 

A = np.array([
    [-2, 1, 0, 0],
    [1, -2, 1, 0],
    [0, 1, -2, 1],
    [0, 0, 1, -2]
], dtype=float)

def f(t, psi):
    y = psi[:4]
    v = psi[4:]
    dy = v
    dv = 10 * A @ y
    return np.concatenate((dy, dv))

psi0 = np.concatenate((x0.flatten(), v0.flatten()))

t_rk, psi_rk = RK2Coup(f, 0.0, psi0, 0.001, 2*np.pi)

# # # Numeric Solution
plt.title(r"Numeric Solution")
for i in range(4):
    plt.plot(t_rk, psi_rk[:, i], label=rf"$x_{i+1}$")

plt.xlabel(r"$t$")
plt.ylabel(r"$y(t)$")
plt.legend(loc="best")

plt.savefig("plot_numeric.png",dpi=300)


# # # Analitic Solution

plt.close()
for i in range(4):
    plt.plot(t, xi(i+1,t,x0,v0), label=rf"$x_{i+1}$")
# x1 = xi(1,t,x0,v0)
# x2 = xi(2,t,x0,v0)
# x3 = xi(3,t,x0,v0)
# x4 = xi(4,t,x0,v0)

# plt.plot(t, x1, label=rf"$x_1$")
# plt.plot(t, x2, label=rf"$x_2$")
# plt.plot(t, x3, label=rf"$x_3$")
# plt.plot(t, x4, label=rf"$x_4$")

plt.title(r"Analytic Solution")
plt.legend(loc="best")
plt.savefig("plot_analiric.png",dpi=300)
# plt.savefig("plot.pgf")