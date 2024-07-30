import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = 3 * x + 7 + np.random.normal(size=x.size)

A = np.vstack([x, np.ones_like(x)]).T
slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"Slope: {slope}, Intercept: {intercept}")

plt.scatter(x, y, label='Data')
plt.plot(x, slope * x + intercept, color='red', label='Fit')
plt.legend()
plt.show()
