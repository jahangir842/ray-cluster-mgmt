import time, math, random

def calculate_pi(samples):
    inside = 0
    for _ in range(samples):
        if math.hypot(random.random(), random.random()) <= 1:
            inside += 1
    return inside

if __name__ == "__main__":
    TOTAL = 500_000_000
    print("Running on 1 CPU Core...")
    
    start = time.time()
    calculate_pi(TOTAL)
    
    print(f"Time: {time.time() - start:.2f} seconds")