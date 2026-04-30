import time
import numpy as np

MATRIX_SIZE = 16384
NUM_TASKS = 300

def matmul(num_tasks, size):
    rng = np.random.default_rng(42)
    A = rng.random((num_tasks, size, size), dtype=np.float32)
    B = rng.random((num_tasks, size, size), dtype=np.float32)
    return np.matmul(A, B)

t0 = time.perf_counter()
C = matmul(NUM_TASKS, MATRIX_SIZE)
t = time.perf_counter() - t0

print(f"Single Machine | numpy BLAS | {MATRIX_SIZE}x{MATRIX_SIZE}\n")
print(f"{'Tasks':<8} {'Time (s)':>10}")
print(f"{NUM_TASKS:<8} {t:>10.2f}")
print(f"\nResult (sum of all outputs): {C.sum():.4f}")