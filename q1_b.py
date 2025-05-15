import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Duffing parameters
lambda0 = 1.0
lambda1 = 0.1
m = 1.0
a = 1.0

# omega0_sq = lmbda0 / m
# alpha = lmbda1 / m

# Duffing system definition
def cubic(t, y):
    x, v = y
    dxdt = v
    dvdt = -2*(-lambda0*x+lambda1*(x**3))*(np.sqrt(a**2 + x**2) - a)*(x/np.sqrt(x**2 + a**2))/m
    return [dxdt, dvdt]

# Time span and resolution
t_span = (0,20)
t_eval = np.linspace(*t_span, 1000)

# Initial condition ranges for x and v
x_vals = np.linspace(-2, 2, 5)   # displacement range
v_vals = np.linspace(-2, 2, 5)   # velocity range

# Plotting phase portrait
plt.figure(figsize=(8, 6))
for x0 in x_vals:
    for v0 in v_vals:
        sol = solve_ivp(cubic, t_span, [x0, v0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        plt.plot(sol.y[0], sol.y[1], lw=1)

# Axes and labels
plt.xlabel("x", fontsize=18)
plt.ylabel("v = dx/dt", fontsize=18)
plt.title("Phase Portrait", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()