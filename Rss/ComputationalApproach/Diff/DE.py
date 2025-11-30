# Example of numerical differential equation solution


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


def RK2(f, t0, y0, h, t_akh):
    t_rk2 = np.arange(t0, t_akh + h, h)
    y_rk2 = np.zeros(len(t_rk2))
    y_rk2[0] = y0

    for i in range(len(t_rk2) - 1):
        t_i = t_rk2[i]
        y_i = y_rk2[i]
        t_ip = t_rk2[i + 1]

        k1 = f(t_i, y_i)
        y_pred = y_i + h * k1
        k2 = f(t_ip, y_pred)

        y_rk2[i + 1] = y_i + 0.5 * h * (k1 + k2)

    return t_rk2, y_rk2
