# Matrix Multiplication — Ray Core Example

Runs 300 × 4096×4096 matrix multiplications and compares single-machine
sequential execution against Ray cluster parallelism.

## Scripts

| File | What it does |
|------|-------------|
| `1-single-machine.py` | Baseline — runs all 300 tasks sequentially on one CPU |
| `2-cluster-auto-sheduling.py` | Ray auto-scheduling — tasks spread across the cluster automatically |
| `2-cluster-manual-sheduling.py` | Ray `SPREAD` strategy — forces even distribution across all nodes |

## Run

```bash
# Baseline (no Ray needed)
python 1-single-machine.py

# On the Ray cluster
ray job submit --address="http://192.168.3.73:8265" --working-dir . \
  -- python 2-cluster-auto-sheduling.py

```

## Expected output (cluster)

```
Ray Cluster | 8 nodes | 268 CPUs | 300 tasks | 4096x4096
Time   : 4.83s
Result : 2097152.0000

  pc1-4500    42 tasks
  pc2-4500    41 tasks
  ...
```

The cluster run should be significantly faster than the single-machine baseline
as tasks execute in parallel across all available CPUs.
