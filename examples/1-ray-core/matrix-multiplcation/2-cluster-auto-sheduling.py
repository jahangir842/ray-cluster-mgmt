import time, socket
import numpy as np
import ray
from collections import Counter

MATRIX_SIZE = 4096
NUM_TASKS = 300

@ray.remote
def matmul_task(task_id):
    A = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float32)
    B = np.full((MATRIX_SIZE, MATRIX_SIZE), 0.5, dtype=np.float32)
    C = np.matmul(A, B)
    # Use float64 for the sum to avoid float32 precision loss on large reductions
    return socket.gethostname(), float(C.astype(np.float64).sum())

# Connect to cluster
ray.init(address="auto", ignore_reinit_error=True)
cpus  = int(ray.cluster_resources().get("CPU", 0))
nodes = sum(1 for n in ray.nodes() if n["Alive"])

t0 = time.perf_counter()
results = ray.get([matmul_task.remote(i) for i in range(NUM_TASKS)])
t = time.perf_counter() - t0

hostnames, values = zip(*results)
dist = Counter(hostnames)

print(f"Ray Cluster | {nodes} nodes | {cpus} CPUs | {MATRIX_SIZE}x{MATRIX_SIZE}\n")
print(f"{'Tasks':<8} {'Time (s)':>10}")
print(f"{NUM_TASKS:<8} {t:>10.2f}\n")

for node, count in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {node:<25} {count:>4} tasks  {'█' * count}")

print(f"\n  Nodes used : {len(dist)} / {nodes}")
print(f"\nResult (sum of all outputs): {sum(values):.4f}")