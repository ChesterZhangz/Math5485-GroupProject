"""Exercise 2 (iv): Euler maximum error and convergence rate."""
from pathlib import Path
import runpy

import numpy as np
import matplotlib.pyplot as plt


u0, T = 1, 2

def u_exact(t):
    return 1 / (2 - np.cos(t))

ns = np.arange(1, 501)
errs = errors(Euler, f, u0, T, ns, u_exact)

fig, ax = plt.subplots()
plot_errors(ax, ns, errs, 'Euler')
ax.set_title("Euler's Method: Maximum Error")
ax.grid(True, which='both')
plt.show()

print('Estimated convergence rate:', rate(ns, errs))
