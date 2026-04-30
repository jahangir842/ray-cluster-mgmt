import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

MATRIX_SIZE = 2048
NUM_CORES = 32

def matmul(seed):
    rng = np.random.default_rng(seed)
    A = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    B = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    return np.matmul(A, B).sum()

print(f"Single Machine | {NUM_CORES} cores | {MATRIX_SIZE}x{MATRIX_SIZE} matrices\n")
print(f"{'Tasks':<8} {'Time (s)':>10} {'Tasks/s':>10} {'Time/task (s)':>15}")
print("-" * 47)

times = {}
for n in [64, 100, 150, 200]:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=NUM_CORES) as ex:
        list(ex.map(matmul, range(n)))
    t = time.perf_counter() - t0
    times[n] = t
    print(f"{n:<8} {t:>10.2f} {n/t:>10.2f} {t/n:>15.4f}")

print("\n--- Final Result ---")
print(f"  Best throughput : {max(n/t for n,t in times.items()):.2f} tasks/s  (at {max(times, key=lambda n: n/times[n])} tasks)")
print(f"  Worst time      : {max(times.values()):.2f}s  (at {max(times, key=times.get)} tasks)")
print(f"  Time scaling    : {times[200]/times[64]:.2f}x slower going 64 → 200 tasks")
print(f"  Bottleneck      : {NUM_CORES} cores  (tasks queue up beyond this)")