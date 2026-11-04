from pathlib import Path
import runpy

import numpy as np
import matplotlib.pyplot as plt


def u_exact(t, u0=1.0):
    return u0 / ((1.0 + u0) - u0 * np.cos(t))

def Heun(f, u0, T, n):
    h = T / n
    t = np.linspace(0.0, T, n + 1)
    u = np.zeros(n + 1)
    u[0] = u0
    for j in range(n):
        k1 = f(t[j], u[j])
        predict = u[j] + h * k1
        k2 = f(t[j + 1], predict)
        u[j + 1] = u[j] + (h / 2.0) * (k1 + k2)
    return u

def RK4(f, u0, T, n):
    h = T / n
    t = np.linspace(0.0, T, n + 1)
    u = np.zeros(n + 1)
    u[0] = u0
    for j in range(n):
        k1 = f(t[j], u[j])
        k2 = f(t[j] + h / 2.0, u[j] + (h / 2.0) * k1)
        k3 = f(t[j] + h / 2.0, u[j] + (h / 2.0) * k2)
        k4 = f(t[j + 1], u[j] + h * k3)
        u[j + 1] = u[j] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return u


ns = np.unique(np.round(10.0 ** np.linspace(0.0, 6.0, 25)).astype(int))

errE2, Ecount = [], []
errH2, Hcount = [], []
errR2, Rcount = [], []
for n in ns:
    g, c = counted(f)
    u = Euler(g, 1.0, 2.0, n)
    errE2.append(max_error(u, u_exact, 2.0))
    Ecount.append(c[0])

    g, c = counted(f)
    u = Heun(g, 1.0, 2.0, n)
    errH2.append(max_error(u, u_exact, 2.0))
    Hcount.append(c[0])

    g, c = counted(f)
    u = RK4(g, 1.0, 2.0, n)
    errR2.append(max_error(u, u_exact, 2.0))
    Rcount.append(c[0])
    
errE2, Ecount = np.array(errE2), np.array(Ecount)
errH2, Hcount = np.array(errH2), np.array(Hcount)
errR2, Rcount = np.array(errR2), np.array(Rcount)

for name, count, err in [
    ("Euler", Ecount, errE2),
    ("Heun", Hcount, errH2),
    ("RK4", Rcount, errR2)]:
    idx = np.where(err <= 1e-6)[0]

    if len(idx) > 0:
        j = idx[0]
        print(f"{name}: {count[j]} evaluations each step")

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(Ecount, errE2, "-o", ms=3, label="Euler")
ax.loglog(Hcount, errH2, "-s", ms=3, label="Heun")
ax.loglog(Rcount, errR2, "-d", ms=3, label="RK4")

ax.axhline(1e-6, color="gray", linestyle="--",
           label=r"Tolerance $10^{-6}$")

ax.set_xlabel("Total number of steps")
ax.set_ylabel(r"Maximum error $e_n$")
ax.legend()
plt.tight_layout()
plt.show()
