"""
Matrix Multiplication Benchmark: Ray Cluster vs Single Machine
Usage: python matrix_benchmark.py
"""

import time
import numpy as np
import ray
from concurrent.futures import ProcessPoolExecutor

MATRIX_SIZE = 2048
NUM_TASKS = 64
NUM_CORES = 32


def single_matmul(seed):
    rng = np.random.default_rng(seed)
    A = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    B = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    return np.matmul(A, B).sum()


@ray.remote
def ray_matmul(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    A = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    B = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    return np.matmul(A, B).sum()


# --- Single Machine ---
print(f"Running {NUM_TASKS} matmuls on single machine ({NUM_CORES} cores)...")
t0 = time.perf_counter()
with ProcessPoolExecutor(max_workers=NUM_CORES) as ex:
    list(ex.map(single_matmul, range(NUM_TASKS)))
single_time = time.perf_counter() - t0
print(f"  Single machine: {single_time:.2f}s")

# --- Ray Cluster ---
ray.init(address="auto", ignore_reinit_error=True)
cpus = int(ray.cluster_resources().get("CPU", 0))
print(f"\nRunning {NUM_TASKS} matmuls on Ray cluster ({cpus} CPUs)...")
t0 = time.perf_counter()
ray.get([ray_matmul.remote(seed) for seed in range(NUM_TASKS)])
cluster_time = time.perf_counter() - t0
print(f"  Ray cluster:    {cluster_time:.2f}s")

# --- Summary ---
speedup = single_time / cluster_time
print(f"\nSpeedup: {speedup:.2f}x ({'cluster faster' if speedup > 1 else 'single faster'})")