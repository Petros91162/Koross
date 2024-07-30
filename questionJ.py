# Find the eigenvalues and eigenvectors of the matrix:

import numpy as np

A = np.array([[4, 1, 1],
              [1, 3, -1],
              [1, -1, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Power Iteration Method


def power_iteration(A, num_iterations):
    # Initialize a random vector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_iterations):
        # Multiply the matrix with the vector
        b_k1 = A @ b_k

        # Normalize the result vector
        b_k = b_k1 / np.linalg.norm(b_k1)

    # Multiply the normalized vector with the matrix to get the eigenvalue
    eigenvalue = (A @ b_k).T @ b_k

    return eigenvalue, b_k


# Qr Algorithm

def qr_algorithm(A, num_iterations):
    Q = np.eye(A.shape[0])

    for _ in range(num_iterations):
        # Compute the QR decomposition of A
        Q, R = np.linalg.qr(A)

        # Update A using the QR decomposition
        A = R @ Q

    # The diagonal elements of A are the eigenvalues
    eigenvalues = np.diag(A)

    return eigenvalues, Q
