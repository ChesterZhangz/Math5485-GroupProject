
from pathlib import Path
import runpy

import numpy as np
import matplotlib.pyplot as plt


mu = 0.5  # illustrative choice; the problem does not specify mu
T = 2.0  # illustrative choice; the problem does not specify T
u0 = 1.0


def f(t, u):
    return mu * u


def u_exact(t):
    return np.exp(mu * t)


ns = np.arange(10, 501)
# The nth entry returned by Euler is the approximation at t = T.
approximations = np.array([Euler(f, u0, T, int(n))[-1] for n in ns])
errs = np.abs(approximations - u_exact(T))

fig, ax = plt.subplots()
ax.loglog(ns, errs, label='Euler error at T', linewidth=2)
if mu * T != 0:
    asymptotic = np.exp(mu * T) * (mu * T) ** 2 / (2 * ns)
    ax.loglog(ns, asymptotic, '--', label=r'Asymptotic $e^{\mu T}(\mu T)^2/(2n)$')
    print(f'Estimated convergence rate: {rate(ns, errs):.6f}')
else:
    print('mu*T = 0: Euler is exact, so the error is zero.')
ax.set_xlabel('n')
ax.set_ylabel(r'$|u_n - e^{\mu T}|$')
ax.set_title('Compound Interest: Euler Convergence')
ax.grid(True, which='both')
ax.legend()
fig.tight_layout()
fig.savefig(Path(__file__).with_name('exercise2-6.png'), dpi=200)
plt.show()

print(f'Exact solution at T: {u_exact(T):.10f}')
print(f'Euler approximation at n=500: {approximations[-1]:.10f}')
