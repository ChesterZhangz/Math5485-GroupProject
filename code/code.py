import numpy as np
import matplotlib.pyplot as plt
import warnings

# Euler's method: u_{j+1} = u_j + h f(t_j, u_j) (see Exercise 2)
def Euler(f, u0, T, n):
    h = T / n
    t = np.linspace(0, T, n + 1)
    u = np.zeros(n + 1)
    u[0] = u0

    for j in range(n):
        u[j + 1] = u[j] + h * f(t[j], u[j])

    return u


# Newton's method for F(x) = 0 (from Lecture 5):
# returns the final iterate and the number of iterations
def Newton(F, F_prime, x1, N=100, tol=1e-12):
    x = x1

    for k in range(1, N + 1):
        x = x - F(x) / F_prime(x)

        if abs(F(x)) < tol:
            return x, k

    warnings.warn(f"Newton did not converge within {N} iterations.")
    return x, N


# Maximum error max_j |u(t_j) - u_j|
# over the mesh points t_j = j T/n
def max_error(u, u_exact, T):
    n = len(u) - 1
    t = np.linspace(0, T, n + 1)

    return np.max(np.abs(u - u_exact(t)))


# Errors e_n = max_j |u_j - u(t_j)| for each n in ns.
# method(f, u0, T, n) is a solver, e.g. Euler,
# or lambda f, u0, T, n: Trapezoid(f, df_du, u0, T, n).
def errors(method, f, u0, T, ns, u_ex):
    return np.array([
        max_error(method(f, u0, T, n), u_ex, T)
        for n in ns
    ])


# A log-log plot of the errors against n.
# Start with fig, ax = plt.subplots(), then add more curves.
def plot_errors(ax, ns, errs, label):
    ax.loglog(
        ns, errs,
        linewidth=2,
        marker='o',
        markersize=3,
        label=label
    )

    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$e_n$")
    ax.legend(loc="lower left")

    return ax


# Estimated rate r in e_n ≈ C n^(-r)
# (least squares fit on the log-log scale,
# using the larger half of the n values)
def rate(ns, errs):
    ns = np.asarray(ns)
    errs = np.asarray(errs)

    k = len(ns) // 2

    # Julia uses 1-based indexing, so ns[k:end]
    # corresponds to Python's ns[k-1:]
    x = np.log(ns[k - 1:])
    y = np.log(errs[k - 1:])

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    return -np.sum(
        (x - x_mean) * (y - y_mean)
    ) / np.sum((x - x_mean) ** 2)


# counted(f) returns (g, count):
# g(t, u) = f(t, u), and count[0] is the
# number of times that g has been called
def counted(f):
    count = [0]

    def g(t, u):
        count[0] += 1
        return f(t, u)

    return g, count


# The model problem:
# u' = f(t, u) = -u² sin(t)
# df_du = ∂f/∂u, needed for Newton's method
def f(t, u):
    return -u**2 * np.sin(t)


def df_du(t, u):
    return -2 * u * np.sin(t)