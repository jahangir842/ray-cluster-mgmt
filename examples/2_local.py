import time, math, random, ray

@ray.remote
def calculate_pi(samples):
    inside = 0
    for _ in range(samples):
        if math.hypot(random.random(), random.random()) <= 1:
            inside += 1
    return inside

if __name__ == "__main__":
    ray.init() # Starts an isolated local instance
    CORES = int(ray.cluster_resources().get('CPU', 1))
    TOTAL = 500_000_000
    print(f"Running on {CORES} Local Cores...")
    
    start = time.time()
    futures = [calculate_pi.remote(TOTAL // CORES) for _ in range(CORES)]
    ray.get(futures)
    
    print(f"Time: {time.time() - start:.2f} seconds")
    ray.shutdown()