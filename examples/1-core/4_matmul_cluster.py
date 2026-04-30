import ray
import time
import numpy as np
import os

# Connect to the existing cluster
ray.init(address="auto", ignore_reinit_error=True)

# 1. Define resources clearly to force the scheduler to wait for memory
# We use 4GB per task as a safe estimate.
@ray.remote(num_cpus=1, memory=4 * 1024 * 1024 * 1024)
def burn_cpu_matrix_remote(size, task_id):
    # This prints the Node ID to the logs so you can verify distribution
    node_id = ray.get_runtime_context().get_node_id()
    print(f"Task {task_id} running on Node: {node_id}")
    
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    result = np.matmul(A, B)
    return np.sum(result)

if __name__ == "__main__":
    # Increased size to make the compute work worth the network cost
    MATRIX_SIZE = 4000 
    NUM_TASKS = 20
    
    print(f"--- Starting Cluster Benchmarking ({NUM_TASKS} tasks) ---")
    start = time.time()
    
    # Fire tasks with task_id for tracking
    futures = [burn_cpu_matrix_remote.remote(MATRIX_SIZE, i) for i in range(NUM_TASKS)]
    
    # Gather results
    results = ray.get(futures)
    
    duration = time.time() - start
    print(f"--- Completed in: {duration:.2f} seconds ---")
    print(f"Final Sum: {sum(results):,.2f}")
    
    ray.shutdown()