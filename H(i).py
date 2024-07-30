import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x, y, x_val):
    """
    Perform Lagrange interpolation to estimate the value at x_val.
    
    Parameters:
    x (array_like): Array of x-coordinates of the data points.
    y (array_like): Array of y-coordinates of the data points.
    x_val (float): The x-coordinate at which to evaluate the interpolated value.

    Returns:
    float: The interpolated value at x_val.
    """
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term = term * (x_val - x[j]) / (x[i] - x[j])
        result += term
    return result

# Given data points
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

# Evaluate at several points
x_vals = np.linspace(1, 4, 100)
y_vals = [lagrange_interpolation(x, y, xv) for xv in x_vals]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Data points', markersize=8, color='red')
plt.plot(x_vals, y_vals, '-', label='Lagrange Interpolation', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Lagrange Polynomial Interpolation')
plt.show()
