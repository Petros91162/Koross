import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the function to fit
def func(x, a, b, c):
    return a * np.exp(b * x) + c

# Generate some data
xdata = np.linspace(0, 4, 50)
ydata = func(xdata, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(xdata))

# Define the cost function for curve fitting
def cost(params, x, y):
    a, b, c = params
    return np.sum((y - func(x, a, b, c))**2)

# Initial guess for the parameters
initial_guess = [1, 1, 1]

# Perform the curve fitting using gradient descent
result = minimize(cost, initial_guess, args=(xdata, ydata), method='BFGS')
popt = result.x

print(f"Fitted parameters: {popt}")

# Plot the data and the fit
plt.scatter(xdata, ydata, label='Data')
plt.plot(xdata, func(xdata, *popt), label='Fit', color='red')
plt.legend()
plt.show()
