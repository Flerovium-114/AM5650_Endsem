import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Define the Rössler system
def rossler_system(t, xyz, a, b, c):
    x, y, z = xyz
    dx_dt = -y - z
    dy_dt = x + a*y
    dz_dt = b + z*(x - c)
    return [dx_dt, dy_dt, dz_dt]

# Parameters
a = 0.1
b = 0.1
c = 14.0

# Initial conditions
x0, y0, z0 = 1.0, 1.0, 1.0
initial_conditions = [x0, y0, z0]

# Time points
t_span = (0, 500)  # Long time span for convergence
t_eval = np.linspace(0, 500, 10000)  # Dense sampling for smooth curves

# Solve the system of ODEs
solution = solve_ivp(
    rossler_system, 
    t_span, 
    initial_conditions, 
    args=(a, b, c), 
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8, 
    atol=1e-8
)

t = solution.t
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]

# Define our function f(x,y,z) mapping to u in [0,1]
# Using a sigmoid-based function to ensure u stays in [0,1]
def f(x, y, z):
    # Simple approach: normalize based on position in phase space
    # Using sigmoid function: 1/(1+e^(-value))
    value = 0.1*(x + y - z/10)  # Linear combination of variables
    return abs(2*np.sin(x*y*z))#*np.sin(y)*np.sin(z))
    #return 1/(1 + np.exp(-value))

# Calculate u from the solution
u = f(x, y, z)

# Discard the first part of the solution as transient
# (first 30% of the data)
transient_cutoff = int(len(t) * 0.3)
t_steady = t[transient_cutoff:]
u_steady = u[transient_cutoff:]

# Create the recurrence plot (u_n-1 vs u_n)
u_n = u_steady[1:]
u_n_minus_1 = u_steady[:-1]

# Create a 3D plot of the Rössler attractor
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(x[transient_cutoff:], y[transient_cutoff:], z[transient_cutoff:], 
         linewidth=0.5, color='blue')
ax1.set_title("Rössler Attractor", fontsize=20)
ax1.set_xlabel("X", fontsize=18)
ax1.set_ylabel("Y", fontsize=18)
ax1.set_zlabel("Z", fontsize=18)
plt.tight_layout()
plt.show()

# === 2. Time series of u(t) ===
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(t_steady, u_steady, linewidth=0.5, color='red')
ax2.set_title("u(t) = f(x, y, z)", fontsize=20)
ax2.set_xlabel("Time (s)", fontsize=18)
ax2.set_ylabel("u", fontsize=18)
ax2.tick_params(axis='both', labelsize=14)
ax2.grid(True)
plt.tight_layout()
plt.show()

# === 3. Recurrence plot (u_{n-1} vs u_n) ===
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111)
ax3.scatter(u_n_minus_1, u_n, s=0.5, color='black')
ax3.set_title("Recurrence Plot", fontsize=20)
ax3.set_xlabel(r'$u_{n-1}$', fontsize=18)
ax3.set_ylabel(r'$u_n$', fontsize=18)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.tick_params(axis='both', labelsize=14)
ax3.grid(True)
ax3.set_aspect('equal')
plt.tight_layout()
plt.show()


# Let's also examine a different function f to see how it affects the recurrence plot
def f2(x, y, z):
    # Another approach: using arctan and normalization
    value = np.arctan(x + y + z) / np.pi + 0.5
    return value

# Calculate u with the second function
u2 = f2(x, y, z)
u2_steady = u2[transient_cutoff:]

# Create the recurrence plot for the second function
u2_n = u2_steady[1:]
u2_n_minus_1 = u2_steady[:-1]

fig2 = plt.figure(figsize=(15, 5))

# First subplot: Time series of u with f2
ax1 = fig2.add_subplot(121)
ax1.plot(t_steady, u2_steady, linewidth=0.5, color='green')
ax1.set_title("u(t) = f2(x,y,z)")
ax1.set_xlabel("Time")
ax1.set_ylabel("u")
ax1.grid(True)

# Second subplot: Recurrence plot with f2
ax2 = fig2.add_subplot(122)
ax2.scatter(u2_n_minus_1, u2_n, s=0.5, color='black')
ax2.set_title("Recurrence Plot (Alternative f)")
ax2.set_xlabel("u_n-1")
ax2.set_ylabel("u_n")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.grid(True)
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()

# Commentary on the recurrence plots:
"""
The recurrence plot shows u_n−1 (previous value) vs u_n (current value).
This plot reveals the structure of the dynamics in a single variable.

Key observations:
1. The points form a distinct pattern rather than filling the space randomly,
   indicating deterministic (not random) behavior
2. The curve-like structures suggest that the mapping from u_n−1 to u_n
   is not a simple function but has multiple branches
3. The bounded nature of the plot reflects our constraint of u ∈ [0,1]
4. The specific shape depends on our choice of function f(x,y,z)

The structure seen in the recurrence plot is characteristic of chaotic systems,
showing a deterministic but complex relationship between consecutive values.
This is directly related to the chaotic nature of the Rössler system itself.
"""