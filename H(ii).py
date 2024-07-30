import numpy as np
import matplotlib.pyplot as plt

def divided_diff(x, y):
    """
    Compute divided difference coefficients for Newton interpolation.
    
    Parameters:
    x (array_like): Array of x-coordinates of the data points.
    y (array_like): Array of y-coordinates of the data points.
    
    Returns:
    np.ndarray: Array of divided difference coefficients.
    """
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
    
    return coef[0, :]

def newton_poly(coef, x_data, x):
    """
    Evaluate the Newton interpolation polynomial at x.
    
    Parameters:
    coef (array_like): Array of divided difference coefficients.
    x_data (array_like): Array of x-coordinates of the data points.
    x (float): The point at which to evaluate the polynomial.
    
    Returns:
    float: The interpolated value at x.
    """
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

# Given data points
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

# Calculate coefficients
coef = divided_diff(x, y)

# Evaluate at several points
x_vals = np.linspace(1, 4, 100)
y_vals = [newton_poly(coef, x, xv) for xv in x_vals]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Data points', markersize=8, color='red')
plt.plot(x_vals, y_vals, '-', label='Newton Interpolation', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Newton Polynomial Interpolation')
plt.show()
