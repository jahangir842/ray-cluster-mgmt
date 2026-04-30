import time
import numpy as np
import ray

MATRIX_SIZE = 2048

@ray.remote
def matmul(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    A = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    B = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    return np.matmul(A, B).sum()

ray.init(address="auto", ignore_reinit_error=True)
cpus = int(ray.cluster_resources().get("CPU", 0))
print(f"Cluster: {cpus} CPUs\n")

print(f"{'Tasks':<10} {'Time (s)':<12} {'Tasks/s':<12} {'Speedup vs 64'}")
print("-" * 50)

baseline = None
for num_tasks in [64, 100, 150, 200]:
    t0 = time.perf_counter()
    ray.get([matmul.remote(seed) for seed in range(num_tasks)])
    elapsed = time.perf_counter() - t0
    tps = num_tasks / elapsed
    if baseline is None:
        baseline = elapsed
    print(f"{num_tasks:<10} {elapsed:<12.2f} {tps:<12.2f} {baseline/elapsed:.2f}x")