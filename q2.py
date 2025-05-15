import numpy as np
import matplotlib.pyplot as plt

# Define the vector field
def f(x, y):
    dxdt = y * (2 * y**2 - 3 * x**2 + (19/9) * x**4)
    dydt = y**2 * (3 * x - (38/9) * x**3) - (4 * x**3 - (28/3) * x**5 + (40/9) * x**7)
    return dxdt, dydt

# Generate a grid of points
x = np.linspace(-2, 2, 30)
y = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x, y)
U, V = f(X, Y)

# Normalize arrows
magnitude = np.sqrt(U**2 + V**2)
U_norm = U / (magnitude + 1e-10)
V_norm = V / (magnitude + 1e-10)

# Plot vector field
plt.figure(figsize=(10, 8))
plt.quiver(X, Y, U_norm, V_norm, color='gray', alpha=0.6)
plt.axhline(0, color='k', lw=1)
plt.axvline(0, color='k', lw=1)

# Plot homoclinic orbit: y² = x² - x⁴
x_vals = np.linspace(-1.35, 1.35, 400)
y1_sq = x_vals**2 - x_vals**4
y1 = np.sqrt(np.maximum(y1_sq, 0))
plt.plot(x_vals, y1, 'b', label=r'$y^2 = x^2 - x^4$')
plt.plot(x_vals, -y1, 'b')

# Plot homoclinic orbit: y² = 2x² - (10/9)x⁴
y2_sq = 2 * x_vals**2 - (10/9) * x_vals**4
y2 = np.sqrt(np.maximum(y2_sq, 0))
plt.plot(x_vals, y2, 'r', label=r'$y^2 = 2x^2 - \frac{10}{9}x^4$')
plt.plot(x_vals, -y2, 'r')

# Plot separatrix directions from origin: slopes ±1, ±√2
s = np.linspace(-2, 2, 100)
plt.plot(s, s, 'k--', label='Slope ±1')
plt.plot(s, -s, 'k--')
plt.plot(s, np.sqrt(2) * s, 'g--', label='Slope ±√2')
plt.plot(s, -np.sqrt(2) * s, 'g--')

# Formatting
plt.title("Phase Portrait with Homoclinic Orbits and Separatrices")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()
