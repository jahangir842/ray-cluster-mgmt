import time
import ray

MATRIX_SIZE = 2048
NUM_TASKS = 200

@ray.remote
def matmul(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    A = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    B = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    return np.matmul(A, B).sum()

ray.init(address="auto", ignore_reinit_error=True)
cpus = int(ray.cluster_resources().get("CPU", 0))
nodes = len(ray.nodes())

t0 = time.perf_counter()
results = ray.get([matmul.remote(seed) for seed in range(NUM_TASKS)])
t = time.perf_counter() - t0

print(f"Ray Cluster | {nodes} nodes | {cpus} CPUs | {MATRIX_SIZE}x{MATRIX_SIZE}\n")
print(f"{'Tasks':<8} {'Time (s)':>10}")
print(f"{NUM_TASKS:<8} {t:>10.2f}")
print(f"\nResult (sum of all outputs): {sum(results):.4f}")