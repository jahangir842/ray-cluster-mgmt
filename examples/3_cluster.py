import time, math, ray

@ray.remote
def burn_cpu(operations):
    total = 0
    for i in range(operations):
        total += math.sqrt(i)
    return total

if __name__ == "__main__":
    ray.init(address='auto') # Connects to the 8-node cluster
    CORES = int(ray.cluster_resources().get('CPU', 1))
    TOTAL = 1000000000
    print(f"Running on {CORES} Cluster Cores...")
    
    start = time.time()
    # Split the 100 million square roots evenly across all 132 cluster cores
    futures = [burn_cpu.remote(TOTAL // CORES) for _ in range(CORES)]
    ray.get(futures)
    
    print(f"Time: {time.time() - start:.2f} seconds")
    ray.shutdown()