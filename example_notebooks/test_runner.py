import multiprocessing as mp
from functools import partial

import numpy as np
import scipy.linalg as linalg

# Load data outside the function
path_A = "/Users/stijndeboer/Projects/PCN/PCNtoolkit/example_notebooks/A_29026a24-8ecc-4ad1-85b5-16d43ece6d8c.npy"
path_X_T = "/Users/stijndeboer/Projects/PCN/PCNtoolkit/example_notebooks/X_T_29026a24-8ecc-4ad1-85b5-16d43ece6d8c.npy"

def solve_equation(A, X_T, x):
    """Worker function that solves the linear equation"""
    return linalg.solve(A, X_T, check_finite=False)

def main():
    # Load data inside the main function
    A = np.load(path_A, allow_pickle=False)
    X_T = np.load(path_X_T, allow_pickle=False)
    
    # Create test data
    a = np.random.rand(10, 2)
    
    # Create partial function with fixed arguments
    worker_fn = partial(solve_equation, A, X_T)
    
    print("running...")
    print(f"NumPy version: {np.__version__}")
    
    # Use context manager for proper cleanup
    with mp.Pool(4) as pool:
        results = pool.map(worker_fn, a)
    
    return results

if __name__ == "__main__":
    results = main()
    print(results)