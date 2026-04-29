import time
import math
import random
import ray

def heavy_computation(samples):
    inside_circle = 0
    for _ in range(samples):
        x, y = random.random(), random.random()
        if math.hypot(x, y) <= 1:
            inside_circle += 1
    return inside_circle

@ray.remote(num_cpus=1)
def distributed_computation(samples):
    return heavy_computation(samples)

if __name__ == "__main__":
    TOTAL_SAMPLES = 500_000_000
    
    # ==========================================
    # 1. RUN SEQUENTIAL (1 Local Core)
    # ==========================================
    print("1. Running Sequentially (1 Core)...")
    start_time = time.time()
    heavy_computation(TOTAL_SAMPLES)
    print(f"Time: {time.time() - start_time:.2f} seconds\n")

    # ==========================================
    # 2. RUN LOCAL PARALLEL (All Local Cores)
    # ==========================================
    print("2. Running Local Parallel...")
    # Calling ray.init() without an address starts a local-only instance
    ray.init() 
    local_cpus = int(ray.cluster_resources().get('CPU', 0))
    print(f"Using {local_cpus} local cores on this machine.")
    
    start_time = time.time()
    samples_per_local_chunk = TOTAL_SAMPLES // local_cpus
    futures = [distributed_computation.remote(samples_per_local_chunk) for _ in range(local_cpus)]
    ray.get(futures)
    print(f"Time: {time.time() - start_time:.2f} seconds\n")
    
    # Shut down the local-only instance
    ray.shutdown() 

    # ==========================================
    # 3. RUN CLUSTER PARALLEL (132 Cores)
    # ==========================================
    print("3. Running Cluster Parallel...")
    # Pointing to 'auto' connects to your 8-node network
    ray.init(address='auto') 
    cluster_cpus = int(ray.cluster_resources().get('CPU', 0))
    print(f"Using {cluster_cpus} cluster cores.")
    
    start_time = time.time()
    samples_per_cluster_chunk = TOTAL_SAMPLES // cluster_cpus
    futures = [distributed_computation.remote(samples_per_cluster_chunk) for _ in range(cluster_cpus)]
    ray.get(futures)
    print(f"Time: {time.time() - start_time:.2f} seconds\n")

    ray.shutdown()