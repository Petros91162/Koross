import numpy as np

# Given coordinates
x_coords = np.array([2.00, 4.25])
y_coords = np.array([7.2, 7.1])

# Point to interpolate
x_to_find = 4.0

# Linear interpolation
x1, x2 = x_coords
y1, y2 = y_coords

# Calculate the slope (y2 - y1) / (x2 - x1)
slope = (y2 - y1) / (x2 - x1)

# Calculate the interpolated y value
y_at_x = y1 + slope * (x_to_find - x1)

# Print the result
print(f"The value of y at x = {x_to_find} is {y_at_x}")
