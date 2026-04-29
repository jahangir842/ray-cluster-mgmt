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
@ray.remote(num_cpus=1)
def distributed_computation(samples):
    return heavy_computation(samples)

if __name__ == "__main__":
    # --- The Heavy Configuration ---
    # 1 Billion samples guarantees the CPUs have to sweat.
    # This eliminates the network "jitter" trap entirely.
    TOTAL_SAMPLES = 1_000_000_000
    
    # Match chunks EXACTLY to your total available cluster cores (132).
    # This ensures 0 idle machines and 0 tasks waiting in a queue.
    NUM_CHUNKS = 132  
    SAMPLES_PER_CHUNK = TOTAL_SAMPLES // NUM_CHUNKS

    print(f"--- Starting HEAVY Benchmark: {TOTAL_SAMPLES:,} Operations ---")
    print(f"Each of the {NUM_CHUNKS} tasks will process {SAMPLES_PER_CHUNK:,} operations.\n")

    # ==========================================
    # RUN 1: Sequential (Local Single Core)
    # ==========================================
    print("1. Running Sequentially (Local Machine Only)...")
    print("   (This might take over 60 seconds. Grab a coffee.)")
    start_time = time.time()

    for _ in range(NUM_CHUNKS):
        heavy_computation(SAMPLES_PER_CHUNK)

    sequential_duration = time.time() - start_time
    print(f"Sequential Execution Time: {sequential_duration:.2f} seconds")

    # ==========================================
    # RUN 2: Distributed (Ray Cluster)
    # ==========================================
    print("\n2. Connecting to Ray Cluster...")
    ray.init(address='auto') 
    
    available_cpus = int(ray.cluster_resources().get('CPU', 0))
    print(f"Cluster connected! Access acquired to {available_cpus} CPUs.")
    print(f"Running Distributed ({NUM_CHUNKS} parallel tasks)...")
    start_time = time.time()

    # Fire all 132 chunks simultaneously
    futures = [distributed_computation.remote(SAMPLES_PER_CHUNK) for _ in range(NUM_CHUNKS)]
    
    # Wait for all chunks to finish
    ray.get(futures)

    distributed_duration = time.time() - start_time
    print(f"Distributed Execution Time: {distributed_duration:.2f} seconds")

    # ==========================================
    # FINAL RESULTS
    # ==========================================
    speedup = sequential_duration / distributed_duration
    print(f"\n--- Final Results ---")
    print(f"The Ray Cluster was {speedup:.2f}x faster than local sequential execution!")

    ray.shutdown()