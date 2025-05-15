import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Chen system definition
def chen_system(t, state, a=35, b=3, c=28):
    x, y, z = state
    dxdt = a * (y - x)
    dydt = (c - a) * x - x * z + c * y
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

# Initial condition and time span
t_span = (0, 50)
t_eval = np.linspace(*t_span, 10000)
initial_state = [1, 1, -5]

# Solve the ODE
solution = solve_ivp(chen_system, t_span, initial_state, t_eval=t_eval, args=(35, 3, 28))

# Extract solutions
x, y, z = solution.y

# Phase portrait
fig = plt.figure(figsize=(14, 10))  # Increased figure size
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, lw=0.5)

# Larger title and labels
ax.set_title("Chen Attractor (Phase Portrait)", fontsize=20)
ax.set_xlabel("x", fontsize=18)
ax.set_ylabel("y", fontsize=18)
ax.set_zlabel("z", fontsize=18)

# Optional: increase tick label sizes
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.show()