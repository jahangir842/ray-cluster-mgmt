import time
import math
import random
import ray

# --- 1. The Heavy Mathematical Workload ---
def heavy_computation(samples):
    """Simulates heavy CPU work by calculating Pi."""
    inside_circle = 0
    for _ in range(samples):
        x, y = random.random(), random.random()
        if math.hypot(x, y) <= 1:
            inside_circle += 1
    return inside_circle

# --- 2. The Ray Task Wrapper ---
# We tell Ray that each chunk requires exactly 1 CPU
@ray.remote(num_cpus=1)
def distributed_computation(samples):
    return heavy_computation(samples)

if __name__ == "__main__":
    # --- Configuration ---
    # 50 Million samples takes a noticeable amount of time sequentially
    TOTAL_SAMPLES = 50_000_000
    # We split the work into 20 separate chunks to distribute
    NUM_CHUNKS = 20  
    SAMPLES_PER_CHUNK = TOTAL_SAMPLES // NUM_CHUNKS

    print(f"--- Starting Benchmark: {TOTAL_SAMPLES:,} Operations ---")

    # ==========================================
    # RUN 1: Sequential (Local Single Core)
    # ==========================================
    print("\n1. Running Sequentially (Local Machine Only)...")
    start_time = time.time()

    # The script waits for each chunk to finish before starting the next
    for _ in range(NUM_CHUNKS):
        heavy_computation(SAMPLES_PER_CHUNK)

    sequential_duration = time.time() - start_time
    print(f"Sequential Execution Time: {sequential_duration:.2f} seconds")

    # ==========================================
    # RUN 2: Distributed (Ray Cluster)
    # ==========================================
    print("\n2. Connecting to Ray Cluster...")
    # 'auto' connects to your running head node daemon
    ray.init(address='auto') 
    
    available_cpus = ray.cluster_resources().get('CPU', 0)
    print(f"Cluster connected! Access acquired to {available_cpus} CPUs.")
    print("Running Distributed (Parallel)...")
    start_time = time.time()

    # .remote() fires all 20 chunks off to the cluster simultaneously 
    futures = [distributed_computation.remote(SAMPLES_PER_CHUNK) for _ in range(NUM_CHUNKS)]
    
    # ray.get() pauses the main script until all 20 chunks are returned
    ray.get(futures)

    distributed_duration = time.time() - start_time
    print(f"Distributed Execution Time: {distributed_duration:.2f} seconds")

    # ==========================================
    # FINAL RESULTS
    # ==========================================
    speedup = sequential_duration / distributed_duration
    print(f"\n--- Final Results ---")
    print(f"The Ray Cluster was {speedup:.2f}x faster than local sequential execution!")

    # Clean up the connection
    ray.shutdown()