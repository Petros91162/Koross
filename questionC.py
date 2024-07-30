# Define the points
x_values = [2.00, 4.25]
y_values = [7.2, 7.1]

# Define the x value for which we want to find y
x_target = 4.0

# Extract points
x0, x1 = x_values
y0, y1 = y_values

# Calculate y using the linear spline formula
y_target = y0 + (y1 - y0) / (x1 - x0) * (x_target - x0)

# Print the result
print(f"The value of y at x = {x_target} is approximately: {y_target:.4f}")
