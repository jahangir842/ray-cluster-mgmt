import time
import numpy as np

MATRIX_SIZE = 4096
NUM_TASKS = 300

def matmul():
    matrix_A = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float64)
    matrix_B = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float64)
    return float(np.matmul(matrix_A, matrix_B).sum())

t0 = time.perf_counter()
results = []
for i in range(NUM_TASKS):
    result = matmul()
    results.append(result)
t = time.perf_counter() - t0

print(f"Single Machine | {NUM_TASKS} tasks | {MATRIX_SIZE}x{MATRIX_SIZE}")
print(f"Time   : {t:.2f}s")
print(f"Result : {sum(results):.4f}")