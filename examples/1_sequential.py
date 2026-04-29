import time, math

def burn_cpu(operations):
    total = 0
    # A simple, heavy loop: Just find the square root of every number
    for i in range(operations):
        total += math.sqrt(i)
    return total

if __name__ == "__main__":
    TOTAL = 100_000_000
    print("Running on 1 CPU Core...")
    
    start = time.time()
    burn_cpu(TOTAL)
    
    print(f"Time: {time.time() - start:.2f} seconds")