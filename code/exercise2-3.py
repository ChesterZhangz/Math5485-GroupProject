import numpy as np
import matplotlib.pyplot as plt
import runpy
from pathlib import Path

helpers = runpy.run_path(str(Path(__file__).with_name('code.py')))
Euler = helpers['Euler']
f = helpers['f']

u0 = 1
T = 2


def u_exact(t):
    return 1 / (2 - np.cos(t))


fig, ax = plt.subplots()
t_exact = np.linspace(0, T, 1000)
ax.plot(t_exact, u_exact(t_exact), 'k-', label='Exact solution')

for n in (5, 10, 100):
    t = np.linspace(0, T, n + 1)
    u = Euler(f, u0, T, n)
    ax.plot(t, u, marker='o', markersize=3, label=f'Euler n={n}')

ax.set_xlabel('t')
ax.set_ylabel('u(t)')
ax.set_title("Euler's Method vs. Exact Solution")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
