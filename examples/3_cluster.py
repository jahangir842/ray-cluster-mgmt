import time, math, ray

@ray.remote
def burn_cpu(start_idx, end_idx):
    total = 0
    for i in range(start_idx, end_idx):
        total += math.sqrt(i)
    return total

if __name__ == "__main__":
    ray.init(address='auto') 
    CORES = int(ray.cluster_resources().get('CPU', 1))
    TOTAL = 1_000_000_000
    CHUNK_SIZE = TOTAL // CORES
    
    print(f"Running on {CORES} Cluster Cores...")
    
    start = time.time()
    futures = []
    
    # Blast the unique ranges across the 8-machine network
    for i in range(CORES):
        chunk_start = i * CHUNK_SIZE
        chunk_end = TOTAL if i == CORES - 1 else (i + 1) * CHUNK_SIZE 
        futures.append(burn_cpu.remote(chunk_start, chunk_end))
    
    results = ray.get(futures)
    final_total = sum(results)
    
    print(f"Time: {time.time() - start:.2f} seconds")
    print(f"Final Total: {final_total:,.2f}")
    ray.shutdown()