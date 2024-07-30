import numpy as npy
def f(x):
 return x**3 + 2*x**2 + x + 1

x = npy.linspace(0, 10, 100)
y = f(x)
dy_dx = npy.gradient(y, x)
print(f"The numerical derivative is: {dy_dx}")
