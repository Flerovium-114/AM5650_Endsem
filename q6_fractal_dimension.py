import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Chen system definition
def chen_system(t, state, a=35, b=3, c=28):
    x, y, z = state
    dxdt = a * (y - x)
    dydt = (c - a) * x - x * z + c * y
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

# 2. Integrate the system
t_span = (0, 50)
t_eval = np.linspace(*t_span, 10000)
initial_state = [1, 1, -5]
sol = solve_ivp(chen_system, t_span, initial_state, t_eval=t_eval)
points = np.array([sol.y[0], sol.y[1], sol.y[2]]).T

# Optionally: downsample for speed
points = points[::10]  # ~1000 points

# 3. Grassberger–Procaccia algorithm
def compute_correlation_integral(data, epsilons):
    N = len(data)
    C = []
    for eps in epsilons:
        count = 0
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(data[i] - data[j]) < eps:
                    count += 1
        C_eps = 2 * count / (N * (N - 1))
        C.append(C_eps)
    return np.array(C)

# 4. Define epsilon values and compute
epsilons = np.logspace(-2, 2, 25)
correlation_integral = compute_correlation_integral(points, epsilons)

# 5. Plot and estimate slope
# Filter out zero or very small correlation integral values
valid = correlation_integral > 0
log_eps = np.log2(epsilons[valid])
log_C = np.log2(correlation_integral[valid])

# log_eps = np.log(epsilons)
# log_C = np.log(correlation_integral)

plt.plot(log_eps, log_C, 'o', label="C(eps)")
plt.xlabel('log(epsilon)')
plt.ylabel('log(C(epsilon))')
plt.title('Grassberger–Procaccia Estimate')
plt.grid(True)
plt.legend()


# 6. Estimate slope (correlation dimension) using linear region
from scipy.stats import linregress

# choose linear region manually or automatically
linear_region = slice(2, 14)  # adjust if needed
slope, intercept, r_value, p_value, std_err = linregress(log_eps[linear_region], log_C[linear_region])
print(f"Estimated Correlation Dimension D2 ≈ {slope:.3f}")

log_eps_fit = log_eps[linear_region]
log_C_fit = slope * log_eps_fit + intercept

# Plot the regression line
# plt.figure(figsize=(10, 5))
plt.plot(log_eps_fit, log_C_fit, 'r-', label=f"Fit: D₂ ≈ {slope:.3f}")
plt.xlabel("log₂(epsilon)")
plt.ylabel("log₂(C(epsilon))")
plt.title("Estimated Correlation Dimension (D₂)")
plt.legend()
plt.grid(True)

plt.show()