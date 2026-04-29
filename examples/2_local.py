import time, math, ray

@ray.remote
def burn_cpu(operations):
    total = 0
    for i in range(operations):
        total += math.sqrt(i)
    return total

if __name__ == "__main__":
    ray.init() # Starts an isolated local instance
    CORES = int(ray.cluster_resources().get('CPU', 1))
    TOTAL = 100_000_000
    print(f"Running on {CORES} Local Cores...")
    
    start = time.time()
    # Split the 100 million square roots evenly across your local cores
    futures = [burn_cpu.remote(TOTAL // CORES) for _ in range(CORES)]
    ray.get(futures)
    
    print(f"Time: {time.time() - start:.2f} seconds")
    ray.shutdown()