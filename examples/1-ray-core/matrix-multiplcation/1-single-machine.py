import time
import numpy as np

MATRIX_SIZE = 4096
NUM_TASKS = 300

def matmul_task(task_id):
    A = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float32)
    B = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float32)
    C = np.matmul(A, B)
    # Use float64 for the sum to avoid float32 precision loss on large reductions
    return float(C.astype(np.float64).sum())

t0 = time.perf_counter()
results = [matmul_task(i) for i in range(NUM_TASKS)]
t = time.perf_counter() - t0

final_sum = sum(results)  # sum identical list of floats, matches Ray's approach

print(f"Single Machine | {MATRIX_SIZE}x{MATRIX_SIZE}\n")
print(f"{'Tasks':<8} {'Time (s)':>10}")
print(f"{NUM_TASKS:<8} {t:>10.2f}")
print(f"\nResult (sum of all outputs): {final_sum:.4f}")