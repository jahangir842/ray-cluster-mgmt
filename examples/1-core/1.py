import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

MATRIX_SIZE = 2048
NUM_CORES = 32
NUM_TASKS = 200

def matmul(seed):
    rng = np.random.default_rng(seed)
    A = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    B = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    return np.matmul(A, B).sum()

t0 = time.perf_counter()
with ProcessPoolExecutor(max_workers=NUM_CORES) as ex:
    results = list(ex.map(matmul, range(NUM_TASKS)))
t = time.perf_counter() - t0

print(f"Single Machine | {NUM_CORES} cores | {MATRIX_SIZE}x{MATRIX_SIZE}\n")
print(f"{'Tasks':<8} {'Time (s)':>10}")
print(f"{NUM_TASKS:<8} {t:>10.2f}")
print(f"\nResult (sum of all outputs): {sum(results):.4f}")