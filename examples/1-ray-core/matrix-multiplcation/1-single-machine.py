import time
import numpy as np

MATRIX_SIZE = 4096
NUM_TASKS = 300

def matmul():
    A = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float64)
    B = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float64)
    return float(np.matmul(A, B).sum())

t0 = time.perf_counter()
results = [matmul() for _ in range(NUM_TASKS)]
t = time.perf_counter() - t0

print(f"Single Machine | {NUM_TASKS} tasks | {MATRIX_SIZE}x{MATRIX_SIZE}")
print(f"Time   : {t:.2f}s")
print(f"Result : {sum(results):.4f}")