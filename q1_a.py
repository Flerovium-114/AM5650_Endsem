import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
lmbda = 1.0   # λ
m = 1.0       # mass
a = 1.0       # half length of string

# System of first-order ODEs: y = [x, v]
def system(t, y):
    x, v = y
    denom = np.sqrt(a**2 + x**2)
    acceleration = - (2 * lmbda / m) * x * (1 - a / denom)
    return [v, acceleration]

# Time span and evaluation points
t_span = (0, 100)
t_eval = np.linspace(*t_span, 2000)

# Create grid of initial conditions
x_vals = np.linspace(0.01, 0.05, 4)
v_vals = np.linspace(0.01, 0.05, 4)

# Plotting phase portrait
plt.figure(figsize=(8, 6))
for x0 in x_vals:
    for v0 in v_vals:
        sol = solve_ivp(system, t_span, [x0, v0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        plt.plot(sol.y[0], sol.y[1], lw=1)

plt.xlabel("x", fontsize=18)
plt.ylabel("v = dx/dt", fontsize=18)
plt.title("Phase Portrait", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()