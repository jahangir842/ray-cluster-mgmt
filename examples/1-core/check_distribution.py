import ray
from collections import Counter

MATRIX_SIZE = 2048
NUM_TASKS = 200

@ray.remote
def matmul(seed):
    import numpy as np, socket
    rng = np.random.default_rng(seed)
    A = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    B = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    return socket.gethostname(), np.matmul(A, B).sum()

ray.init(address="auto", ignore_reinit_error=True)
results = ray.get([matmul.remote(seed) for seed in range(NUM_TASKS)])

hostnames, values = zip(*results)
distribution = Counter(hostnames)

print("Task distribution across nodes:\n")
for node, count in sorted(distribution.items(), key=lambda x: -x[1]):
    bar = "█" * count
    print(f"  {node:<20} {count:>4} tasks  {bar}")

print(f"\n  Total nodes used : {len(distribution)}")
print(f"  Total tasks      : {NUM_TASKS}")