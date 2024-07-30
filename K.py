import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return x**2 + y**2 - x*y + x - y + 1

def grad_f(x, y):
    df_dx = 2*x - y + 1
    df_dy = 2*y - x - 1
    return [df_dx, df_dy]

def gradient_descent(starting_point, learning_rate, num_iterations):
    point = list(starting_point)
    points = [point.copy()]
    
    for _ in range(num_iterations):
        grad = grad_f(point[0], point[1])
        point[0] -= learning_rate * grad[0]
        point[1] -= learning_rate * grad[1]
        points.append(point.copy())
    
    return points

# Parameters
starting_point = [0, 0]
learning_rate = 0.1
num_iterations = 100

# Perform Gradient Descent
points = gradient_descent(starting_point, learning_rate, num_iterations)

# Extract points for plotting
x_points = [point[0] for point in points]
y_points = [point[1] for point in points]

# Create a grid of x and y values
x = np.linspace(-1, 2, 400)
y = np.linspace(-1, 2, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plot the results
plt.contour(X, Y, Z, levels=50)
plt.plot(x_points, y_points, 'r.-')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
