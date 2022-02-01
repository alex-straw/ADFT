import numpy as np

def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    N = M.shape[1]  # N=5 --> size of matrix
    v = np.random.rand(N, 1)  # Column of random numbers from 0 to 1
    norm_v = np.linalg.norm(v, 1)  # This is a single value
    v = v / norm_v  # Divide random numbers by the np.linalg.norm of the matrix


    # 1. Multiply M by the damping factor
    # 2. Add (1-d) / N --> i.e., (1-0.85) / 5)
    M_hat = (d * M + (1 - d) / N)

    for i in range(num_iterations):
        v = M_hat @ v  # Multiply m_hat by the random and normalised matrix - Loop for so many iterations
    return v

adj = np.array([[0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0]])

# This takes each column and divides each value by the sum of that particular column: axis = 0.  Still remains 5x5
M = adj/adj.sum(axis=0, keepdims=1)

v = pagerank(M, 100, 0.85)

print(v)