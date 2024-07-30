import numpy as np
def f(x):
    return x**3 + 2*x**2 + x + 1

x = np.linspace(0, 1, 1000)
y = f(x)
integral = np.trapz(y, x)
print(f"The result of the numerical integration is: {integral}")
