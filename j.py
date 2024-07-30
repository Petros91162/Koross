import numpy as np

def power_iteration(A, num_simulations: int):
    # Choose a random vector to start with
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        
        # Re normalize the vector
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)
    eigenvector = b_k
    
    return eigenvalue, eigenvector

# Define the matrix
A = np.array([[4, 1, 1], 
              [1, 3, -1], 
              [1, -1, 2]])

eigenvalue, eigenvector = power_iteration(A, 1000)
print("Eigenvalue:", eigenvalue)
print("Eigenvector:", eigenvector)
