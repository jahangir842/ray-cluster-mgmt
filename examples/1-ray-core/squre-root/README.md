# Square Root — Ray Core Example

Computes `sqrt(i)` for `i` in `0..1,000,000,000` and compares single-core
sequential execution against Ray cluster parallelism.

The cluster script splits the range into equal chunks — one chunk per available
CPU — and runs them in parallel across all nodes.

## Scripts

| File | What it does |
|------|-------------|
| `1_sqrt_single_machine.py` | Baseline — processes all 1B values on a single CPU core |
| `2_sqrt_cluster.py` | Ray cluster — splits work across all CPUs on all nodes |

## Run

```bash
# Baseline (no Ray needed)
python 1_sqrt_single_machine.py

# On the Ray cluster
ray job submit --address="http://192.168.3.73:8265" --working-dir . \
  -- python 2_sqrt_cluster.py
```

## Expected output

```
# Single machine
Running on 1 CPU Core...
Time: 312.45 seconds
Final Total: 21,081,851,083,600.00

# Cluster (268 CPUs)
Running on 268 Cluster Cores...
Time: 1.18 seconds
Final Total: 21,081,851,083,600.00
```

Both produce the same `Final Total` — the cluster just finishes ~250× faster.
