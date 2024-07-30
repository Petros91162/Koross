import numpy as np
import matplotlib.pyplot as plt

def cubic_spline_interpolation(x, y, x_new):
    n = len(x)
    a = y
    h = np.diff(x)
    
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    A[0, 0] = 1
    A[-1, -1] = 1
    
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * (a[i+1] - a[i]) / h[i] - 3 * (a[i] - a[i-1]) / h[i-1]
    
    c = np.linalg.solve(A, b)
    
  
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    
    for i in range(n-1):
        b[i] = (a[i+1] - a[i]) / h[i] - h[i] * (c[i+1] + 2*c[i]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    
   
    y_new = np.zeros_like(x_new)
    
    for i in range(len(x_new)):
        for j in range(n-1):
            if x[j] <= x_new[i] <= x[j+1]:
                dx = x_new[i] - x[j]
                y_new[i] = a[j] + b[j]*dx + c[j]*dx**2 + d[j]*dx**3
    
    return y_new

x = np.linspace(0, 10, 10)
y = np.sin(x)

x_new = np.linspace(0, 10, 100)
y_new = cubic_spline_interpolation(x, y, x_new)

plt.scatter(x, y, label='Data')
plt.plot(x_new, y_new, label='Spline', color='red')
plt.legend()
plt.show()
