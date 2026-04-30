import time
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from collections import Counter

MATRIX_SIZE = 2048
NUM_TASKS = 300

@ray.remote
def matmul(seed):
    import numpy as np, socket
    rng = np.random.default_rng(seed)
    A = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    B = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    return socket.gethostname(), float(np.matmul(A, B).sum())

ray.init(address="auto", ignore_reinit_error=True)
nodes = [n for n in ray.nodes() if n["Alive"]]
cpus  = int(ray.cluster_resources().get("CPU", 0))
print(f"Cluster: {len(nodes)} nodes | {cpus} CPUs\n")

# SPREAD strategy: force Ray to distribute across all nodes
spread_matmul = matmul.options(
    scheduling_strategy="SPREAD"
)

t0 = time.perf_counter()
results = ray.get([spread_matmul.remote(seed) for seed in range(NUM_TASKS)])
t = time.perf_counter() - t0

hostnames, values = zip(*results)
dist = Counter(hostnames)

print("Task distribution:\n")
for node, count in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {node:<20} {count:>4} tasks  {'█' * count}")

print(f"\n  Nodes used : {len(dist)} / {len(nodes)}")
print(f"  Time (s)   : {t:.2f}")
print(f"  Result     : {sum(values):.4f}")