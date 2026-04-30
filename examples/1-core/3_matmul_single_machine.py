import time
import numpy as np

def burn_cpu_matrix(size, task_id):
    print(f"Processing Task {task_id} sequentially...")
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    result = np.matmul(A, B)
    return np.sum(result)

if __name__ == "__main__":
    MATRIX_SIZE = 4000 
    NUM_TASKS = 20
    
    print(f"--- Starting Sequential Benchmarking ({NUM_TASKS} tasks) ---")
    start = time.time()
    
    # Run the 20 tasks sequentially in a standard loop
    results = []
    for i in range(NUM_TASKS):
        res = burn_cpu_matrix(MATRIX_SIZE, i)
        results.append(res)
    
    duration = time.time() - start
    print(f"--- Completed in: {duration:.2f} seconds ---")
    print(f"Final Sum: {sum(results):,.2f}")