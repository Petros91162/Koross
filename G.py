import numpy as np

def trapezoidal_rule(func, a, b, n):
    """
    Compute the integral of a function using the trapezoidal rule.
    
    Parameters:
    func : function
        The function to integrate.
    a : float
        The start of the interval.
    b : float
        The end of the interval.
    n : int
        The number of sub-intervals.
    
    Returns:
    float
        The approximate integral of the function.
    """
    x = np.linspace(a, b, n+1)
    y = func(x)
    h = (b - a) / n
    integral = (h/2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral

# Define a sample function to integrate
def sample_func(x):
    return x**2

# Set the interval and number of sub-intervals
a = 0
b = 1
n = 100

# Compute the integral
integral = trapezoidal_rule(sample_func, a, b, n)
print(f"The integral of the function from {a} to {b} is approximately {integral:.6f}")

# For visualization
import matplotlib.pyplot as plt

x = np.linspace(a, b, 1000)
y = sample_func(x)
plt.plot(x, y, label='f(x) = x^2')

# Trapezoids
x_trap = np.linspace(a, b, n+1)
y_trap = sample_func(x_trap)
plt.fill_between(x_trap, y_trap, alpha=0.3, label='Trapezoids')

plt.title('Trapezoidal Rule Integration')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
