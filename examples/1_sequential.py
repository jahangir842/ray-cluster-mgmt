import time, math

def burn_cpu(start_idx, end_idx):
    total = 0
    for i in range(start_idx, end_idx):
        total += math.sqrt(i)
    return total

if __name__ == "__main__":
    TOTAL = 1_000_000_000
    print("Running on 1 CPU Core...")
    
    start = time.time()
    # Process the entire block from 0 to 1 Billion
    final_total = burn_cpu(0, TOTAL)
    
    print(f"Time: {time.time() - start:.2f} seconds")
    print(f"Final Total: {final_total:,.2f}")