from pathlib import Path
import runpy

import numpy as np
import matplotlib.pyplot as plt

helpers = runpy.run_path(str(Path(__file__).with_name('code.py')))
Euler = helpers['Euler']
Newton = helpers['Newton']
f = helpers['f']
df_du= helpers['df_du']



def Trapezoid(f, df_du, u0, T, n):
    h = T / n
    t = np.linspace(0.0, T, n + 1)
    u = np.zeros(n + 1)
    u[0] = u0

    for j in range(n):
        uj = u[j]
        v1 = uj + h * f(t[j], uj)

        def F(v):
            return v - uj - (h / 2.0) * (f(t[j], uj) + f(t[j + 1], v))

        def dF(v):
            return 1.0 - (h / 2.0) * df_du(t[j + 1], v)

        u[j + 1], _ = Newton(F, dF, v1)

    return u


def u_exact(t, u0=1.0):
    return u0 / ((1.0 + u0) - u0 * np.cos(t))

u0, T, n = 1.0, 2.0, 5

u_E = Euler(f, u0, T, n)
u_T = Trapezoid(f, df_du, u0, T, n)

tt = np.linspace(0.0, T, 1000)
tmesh = np.linspace(0.0, T, n + 1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tt, u_exact(tt, u0), lw=2, label="exact")
ax.plot(tmesh, u_E, "-o", ms=3, lw=1.5, label="Euler")
ax.plot(tmesh, u_T, "-s", ms=3, lw=1.5, label="Trapezoid")
ax.set_xlabel("t")
ax.set_ylabel("u")
ax.set_title("Comparison of Euler and Trapezoid Methods, n=5")
ax.legend()
plt.tight_layout()
plt.show()