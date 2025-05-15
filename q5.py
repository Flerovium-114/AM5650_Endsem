import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------
# Logistic Map Return Map
# -------------------------
def logistic_map(alpha, u0=0.5, N=200):
    u = np.zeros(N)
    u[0] = u0
    for n in range(1, N):
        u[n] = alpha * u[n-1] * (1 - u[n-1])
    return u

def plot_return_map_logistic(alpha):
    u = logistic_map(alpha)
    plt.figure()
    plt.plot(u[:-1], u[1:], 'bo', markersize=5)
    plt.ylabel(r'$u_n$', fontsize = 18)
    plt.xlabel(r'$u_{n-1}$', fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.title(f'Logistic Map Return Map (α = {alpha})', fontsize = 20)
    plt.grid(True)
    plt.show()

# -------------------------
# Exact Chaotic Solution
# -------------------------
def plot_exact_chaotic_solution(N=200):
    n = np.arange(N)
    u = np.sin(2**n)**2
    plt.figure()
    plt.plot(u[:-1], u[1:], 'r.', markersize=5)
    plt.ylabel(r'$u_n$', fontsize = 18)
    plt.xlabel(r'$u_{n-1}$', fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.title(r'Exact Chaotic Solution: $u_n = \sin^2(2^n)$', fontsize = 20)
    plt.grid(True)
    plt.show()

# -------------------------
# Duffing Equation
# -------------------------
def duffing(t, y, k, omega, Gamma):
    x, v = y
    dxdt = v
    dvdt = -k * v + x - x**3 + Gamma * np.cos(omega * t)
    return [dxdt, dvdt]

def simulate_duffing(Gamma, k=0.3, omega=1.2, tmax=5000):
    def rhs(t, y): return duffing(t, y, k, omega, Gamma)
    t_span = (0, tmax)
    t_eval = np.linspace(*t_span, int(tmax * 10))  # fine resolution
    sol = solve_ivp(rhs, t_span, [0.0, 0.0], t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0]

def plot_return_map_duffing(Gamma):
    t, x = simulate_duffing(Gamma)
    T = 2 * np.pi / 1.2  # one forcing period
    times = np.arange(0, 5000, T)  # skip transients
    x_samples = np.interp(times, t, x)
    
    plt.figure()
    plt.plot(x_samples[:-1], x_samples[1:], 'g.', markersize=5)
    plt.ylabel(r'$x_n$', fontsize = 18)
    plt.xlabel(r'$x_{n-1}$', fontsize = 18)
    plt.title(f'Duffing Return Map (Γ = {Gamma})', fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.grid(True)
    plt.show()

# -------------------------
# Run Plots
# -------------------------

# Logistic map return maps
plot_return_map_logistic(alpha=2.8)
plot_return_map_logistic(alpha=3.4)
plot_return_map_logistic(alpha=3)

# Exact chaotic solution
plot_exact_chaotic_solution()

# Duffing return maps for different Γ
for Gamma in [0.2, 0.28, 0.29, 0.37, 0.5]:
    plot_return_map_duffing(Gamma)
