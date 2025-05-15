import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lorenz system
def lorenz(t, state, a, b, c):
    x, y, z = state
    dxdt = a * (y - x)
    dydt = b * x - y - x * z
    dzdt = x * y - c * z
    return [dxdt, dydt, dzdt]

# Parameters
a = 10
c = 8/3
t_span = (0, 1000)
#t_eval = np.linspace(*t_span, 10000)

# Try different b values
for b in [100.5, 120, 166.0, 166.1]:
    x_fp = np.sqrt(c*(b-1))
    y_fp = x_fp
    z_fp = b - 1
    state0 = np.array([x_fp, y_fp, z_fp]) + 0.001 
    sol = solve_ivp(
    lorenz,
    t_span,
    state0,
    args=(a, b, c),
    rtol=1e-9,
    atol=1e-9
    )

    # Plot in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
    ax.set_title(f"Lorenz Attractor: b = {b}", fontsize = 20)
    ax.set_xlabel("x", fontsize = 18)
    ax.set_ylabel("y", fontsize = 18)
    ax.set_zlabel("z", fontsize = 18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)
    plt.tight_layout()
    plt.show()

    # Time series
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, sol.y[0], label='x(t)')
    plt.title(f"x(t) Time Series for b = {b}", fontsize = 20)
    plt.xlabel("Time", fontsize = 18)
    plt.ylabel("x", fontsize = 18)
    plt.grid(True)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend()
    plt.tight_layout()
    plt.show()
