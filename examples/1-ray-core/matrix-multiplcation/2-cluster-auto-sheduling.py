import time, socket
import numpy as np
import ray
from collections import Counter

MATRIX_SIZE = 4096
NUM_TASKS = 300

@ray.remote
def matmul():
    matrix_A = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float64)
    matrix_B = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float64)
    return socket.gethostname(), float(np.matmul(matrix_A, matrix_B).sum())

ray.init(address="auto")
nodes = sum(1 for n in ray.nodes() if n["Alive"])
cpus  = int(ray.cluster_resources().get("CPU", 0))

t0 = time.perf_counter()
results = ray.get([matmul.remote() for _ in range(NUM_TASKS)])
t = time.perf_counter() - t0

hostnames, values = zip(*results)
dist = Counter(hostnames)

print(f"Ray Cluster | {nodes} nodes | {cpus} CPUs | {NUM_TASKS} tasks | {MATRIX_SIZE}x{MATRIX_SIZE}")
print(f"Time   : {t:.2f}s")
print(f"Result : {sum(values):.4f}")
print()
for node, count in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {node:<25} {count:>4} tasks")