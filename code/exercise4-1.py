def Trapezoid_with_counts(f, df_du, u0, T, n, tol=1e-12):
    h = T / n
    t = np.linspace(0, T, n + 1)
    u = np.zeros(n + 1)
    u[0] = u0
    counts = np.zeros(n, dtype=int)

    for j in range(n):
        fj = f(t[j], u[j])

        def Phi(v):
            return v - u[j] - h / 2 * (
                fj + f(t[j + 1], v)
            )

        def Phi_prime(v):
            return 1 - h / 2 * df_du(t[j + 1], v)

        v1 = u[j] + h * fj

        u[j + 1], counts[j] = Newton(Phi, Phi_prime, v1, tol=tol)

    return u, counts


def Trapezoid_fixed(f, u0, T, n, tol=1e-12, maxiter=10000):
    h = T / n
    t = np.linspace(0, T, n + 1)
    u = np.zeros(n + 1)
    u[0] = u0
    counts = np.zeros(n, dtype=int)

    for j in range(n):
        fj = f(t[j], u[j])

        def G(v):
            return u[j] + h / 2 * (
                fj + f(t[j + 1], v)
            )

        v = u[j] + h * fj

        for k in range(1, maxiter + 1):
            v = G(v)

            if abs(v - G(v)) < tol:
                u[j + 1] = v
                counts[j] = k
                break
        else:
            raise RuntimeError(f"Fixed-point iteration failed at step {j}"
            )

    return u, counts


#Comparison
u0, T, n = 1.0, 2.0, 20
tol = 1e-12

u_newton, count_newton = Trapezoid_with_counts(
    f, df_du, u0, T, n, tol=tol
)

u_fixed, count_fixed = Trapezoid_fixed(
    f, u0, T, n, tol=tol
)

print("Newton total iterations:", np.sum(count_newton))
print("Fixed-point total iterations:", np.sum(count_fixed))
print("Maximum difference:", np.max(np.abs(u_newton - u_fixed)))

# Optional: compare iteration counts at individual steps
fig, ax = plt.subplots(figsize=(8, 4))
steps = np.arange(1, n + 1)

ax.plot(steps, count_newton, "o-", label="Newton")
ax.plot(steps, count_fixed, "s-", label="Fixed-point")
ax.set_xlabel("Step j")
ax.set_ylabel("Number of iterations")
ax.set_title("Newton vs Fixed-Point Iteration")
ax.legend()
ax.grid(alpha=0.3)

plt.show()
