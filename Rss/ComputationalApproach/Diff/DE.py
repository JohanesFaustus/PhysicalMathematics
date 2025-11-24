# Example of numerical differential equation solution


def linearconv(nx):
    dx = 2 / (nx - 1)
    nt = 20  # nt is the number of timesteps we want to calculate
    c = 1
    sigma = 0.5

    dt = sigma * dx

    u = numpy.ones(nx)
    u[int(0.5 / dx) : int(1 / dx + 1)] = 2

    un = numpy.ones(nx)

    for n in range(nt):  # iterate through time
        un = u.copy()  ##copy the existing values of u into un
        for i in range(1, nx):
            u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1])


def linearconv2d(nx, nt, u):
    dx = 2 / (nx - 1)
    dy = dx
    c = 1
    sigma = 0.5
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


def diffus(u, nt, CFL):
    nx = len(u)
    dx = 2 / (nx - 1)
    nu = 0.3
    sigma = CFL
    dt = sigma * dx**2 / nu
    un = numpy.zeros(nx)
    for i in range(1, nt):
        un = u.copy()
        for j in range(1, nx - 1):
            u[j] = un[j] + nu * dt / dx**2 * (u[j + 1] - 2 * u[j] + u[j - 1])
    return u


def diffus_2D(u, nt, alpha):
    nx = len(u)
    ny = len(u[0])
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    un = numpy.zeros_like(u)
    for i in range(nt):
        un = u.copy()
        for j in range(1, nx - 1):
            for k in range(1, ny - 1):
                u[j, k] = un[j, k] + alpha * (
                    un[j + 1, k]
                    - 2 * un[j, k]
                    + un[j - 1, k]
                    + un[j, k + 1]
                    - 2 * un[j, k]
                    + un[j, k - 1]
                )
    return u


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

