import ray
import time
import numpy as np

# 1. Initialize Ray. 
# In your K8s cluster, use 'auto' to connect to the existing head.
ray.init(address="auto", ignore_reinit_error=True)

@ray.remote(num_cpus=1) # Tell Ray this task needs 1 CPU core
def burn_cpu_matrix_remote(size):
    # This runs on the worker nodes
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    result = np.matmul(A, B)
    return np.sum(result)

if __name__ == "__main__":
    # Use a safe size (5000 requires ~100MB per matrix)
    MATRIX_SIZE = 20000 
    NUM_TASKS = 20 # Run 20 tasks distributed across the cluster
    
    print(f"Distributing {NUM_TASKS} tasks across the cluster...")
    start = time.time()
    
    # 2. Fire and forget: Launch tasks in parallel
    # This does not run the code; it submits it to the Ray scheduler
    futures = [burn_cpu_matrix_remote.remote(MATRIX_SIZE) for _ in range(NUM_TASKS)]
    
    # 3. Gather results: This blocks until all tasks complete
    results = ray.get(futures)
    
    duration = time.time() - start
    print(f"Total distributed time: {duration:.2f} seconds")
    print(f"Finished {NUM_TASKS} tasks. Final sum of all results: {sum(results):,.2f}")
    
    # Shutdown is not strictly necessary in a cluster environment, 
    # but good practice for local cleanup
    ray.shutdown()