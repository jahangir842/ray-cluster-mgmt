import time
import numpy as np

def burn_cpu_matrix(size):
    """
    Creates two large random matrices and multiplies them.
    This is much more taxing on the CPU than a sqrt loop.
    """
    # Create two square matrices of size x size
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    # Perform matrix multiplication
    # C = A * B (using the @ operator for dot product)
    result = np.matmul(A, B)
    return np.sum(result)

if __name__ == "__main__":
    # Adjust MATRIX_SIZE to change the 'heat' 
    # 5000x5000 requires ~75MB of RAM and significant CPU work
    MATRIX_SIZE = 5000 
    
    print(f"Running Matrix Multiplication ({MATRIX_SIZE}x{MATRIX_SIZE})...")
    
    start = time.time()
    
    # Run the multiplication
    final_sum = burn_cpu_matrix(MATRIX_SIZE)
    
    duration = time.time() - start
    print(f"Time: {duration:.2f} seconds")
    print(f"Result Sum: {final_sum:,.2f}")