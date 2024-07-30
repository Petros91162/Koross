import numpy as np


def lagrange_interpolation(x_points, y_points, x):
    """
    Compute the Lagrange polynomial interpolation at a given x.
    
    Parameters:
    x_points : list or array
        The x-coordinates of the data points.
    y_points : list or array
        The y-coordinates of the data points.
    x : float
        The x value at which to evaluate the polynomial.
    
    Returns:
    float
        The interpolated value at x.
    """
    n = len(x_points)
    result = 0.0

    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term

    return result


# Example usage
x_data = [1, 2, 3, 4]
y_data = [1, 4, 9, 16]
x_value = 2.5
lagrange_result = lagrange_interpolation(x_data, y_data, x_value)
print(f"Lagrange interpolation at x={x_value}: {lagrange_result:.4f}")








def divided_difference(x_points, y_points):
    """
    Compute the divided difference table.
    
    Parameters:
    x_points : list or array
        The x-coordinates of the data points.
    y_points : list or array
        The y-coordinates of the data points.
    
    Returns:
    list
        The list of coefficients for the Newton polynomial.
    """
    n = len(x_points)
    coeffs = np.copy(y_points).astype(float)

    for j in range(1, n):
        for i in range(n - j):
            coeffs[i] = (coeffs[i + 1] - coeffs[i]) / \
                (x_points[i + j] - x_points[i])

    return coeffs


def newton_interpolation(x_points, y_points, x):
    """
    Compute the Newton polynomial interpolation at a given x.
    
    Parameters:
    x_points : list or array
        The x-coordinates of the data points.
    y_points : list or array
        The y-coordinates of the data points.
    x : float
        The x value at which to evaluate the polynomial.
    
    Returns:
    float
        The interpolated value at x.
    """
    coeffs = divided_difference(x_points, y_points)
    n = len(x_points)
    result = coeffs[0]
    product = 1.0

    for i in range(1, n):
        product *= (x - x_points[i - 1])
        result += coeffs[i] * product

    return result


# Example usage
newton_result = newton_interpolation(x_data, y_data, x_value)
print(f"Newton interpolation at x={x_value}: {newton_result:.4f}")
