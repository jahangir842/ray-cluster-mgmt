import time, math, ray

@ray.remote
def burn_cpu(operations):
    total = 0
    for i in range(operations):
        total += math.sqrt(i)
    return total

if __name__ == "__main__":
    ray.init(address='auto') 
    CORES = int(ray.cluster_resources().get('CPU', 1))
    TOTAL = 100_000_000
    CHUNK_SIZE = TOTAL // CORES
    
    print(f"Running on {CORES} Cluster Cores...")
    
    start = time.time()
    # Blast the exact same chunk size across all 8 machines
    futures = [burn_cpu.remote(CHUNK_SIZE) for _ in range(CORES)]
    
    results = ray.get(futures)
    final_total = sum(results)
    
    print(f"Time: {time.time() - start:.2f} seconds")
    print(f"Final Total: {final_total:,.2f}")
    ray.shutdown()