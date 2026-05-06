# Ray Cluster: Developer Guidelines

Welcome to the Ray Cluster. This 8-node environment provides **268 CPUs**, **8 GPUs**, and massive shared memory specifically designed for heavy, distributed Python workloads.

To ensure the cluster remains stable and resource contention is minimized, all developers must adhere to the following guidelines when writing and submitting Ray jobs.

---

## 1. Connecting to the Cluster

Do not run your heavy Python execution loops directly on the head node's OS. Instead, your Python scripts should connect to the existing cluster.

At the top of your Python driver script, initialize the connection. If you are running the script from within a node on the cluster, use `'auto'`:

```python
import ray

# Connect to the existing cluster
ray.init(address='auto')

# Verify connection
print(ray.cluster_resources())
```

> **Note:** If you are connecting remotely from your local laptop (using Ray Client), use the `ray://<head-node-ip>:10001` format instead.

---

## 2. Resource Allocation (The Golden Rule)

Ray uses a logical resource accounting system. **You must explicitly declare the resources your tasks need.** If you do not, Ray defaults to 1 CPU per task and 0 GPUs, which can quickly overwhelm a single node or leave your GPUs sitting idle.

### Allocating CPUs and GPUs

Use the `@ray.remote` decorator to define exactly what your function requires:

```python
# GOOD: Explicitly requesting 1 GPU and 4 CPUs
@ray.remote(num_gpus=1, num_cpus=4)
def process_heavy_tensor(data):
    pass

# BAD: No resources defined. Ray assumes 1 CPU, 0 GPUs.
@ray.remote
def process_heavy_tensor_bad(data):
    pass
```

### Fractional GPUs

If team members are working on smaller inference models that do not require an entire GPU, you can slice the GPU logically to allow multiple tasks to share it:

```python
# Allows 4 of these to run concurrently on a single GPU
@ray.remote(num_gpus=0.25)
def small_inference_task(data):
    pass
```

---

## 3. Memory Management & The Object Store

Our cluster relies heavily on `object_store_memory` (Zero-Copy shared memory) for speed.

**Avoid Anti-Patterns:** Do not pass massive arrays (like 10 GB datasets) as direct arguments to multiple tasks inside a loop. This creates duplicate copies in standard RAM and will crash the node.

**Use `ray.put()`:** If multiple tasks need to read the same massive dataset, place it in the object store once and pass the lightweight reference to your tasks:

```python
# 1. Put the massive dataset into the shared Object Store once
massive_data_ref = ray.put(load_massive_dataset())

# 2. Pass the reference (ObjectRef) to the workers, NOT the actual data
results = [worker_task.remote(massive_data_ref, i) for i in range(10)]
```

---

## 4. Job Submission

There are two ways to run your Ray scripts on the cluster. The method you choose determines where your logs appear and how your job is tracked in the dashboard.

### Method A — Direct Execution (Quick Testing Only)

Run your script directly from any node on the cluster:

```bash
python my_script.py
```

> ⚠️ **Limitation:** Logs are only visible in your terminal. The Dashboard Logs tab will show: *"Driver logs are only available when submitting jobs via the Job Submission API or CLI."*

Use this method only for quick connectivity tests, not for training runs or long workloads.

### Method B — Ray Job CLI (Recommended)

Submit your script as a proper Ray job. This enables full observability — your `print()` output, errors, and metrics will all appear live in the Dashboard under **Jobs → your Job ID → Logs**:

```bash
ray job submit \
    --address="http://192.168.3.73:8265" \
    --working-dir . \
    -- python my_script.py
```

### Useful CLI Commands

```bash
# List all jobs and their status
ray job list

# Stream logs of a running job live
ray job logs <job_id> --follow

# Stop a running job
ray job stop <job_id>
```

### Finding Your Logs in the Dashboard

Once submitted via CLI, navigate to `http://192.168.3.73:8265` and follow these steps:

1. Click **Jobs** in the top navigation bar
2. Find your job by Job ID or entrypoint script name
3. Click on the job to open its detail page
4. Scroll down to the **Logs** section and click the **Driver** tab
5. Your `print()` output — including epoch and batch progress — will appear here live

---

## 5. Observability & Dashboard Access

Navigate to `http://192.168.3.73:8265` in your browser. You must be on the internal network or connected to the office VPN.

What to monitor:

- **Jobs View:** Track the execution of your driver scripts. See if your job is running, pending, or failed.
- **Logs View:** Click any worker to view `stdout` and `stderr` directly in the browser — no need to hunt log files across 8 machines.
- **Metrics & Demands:** Watch memory and GPU utilization in real time. If your task is stuck in *Pending Demands* indefinitely, you have requested resources the cluster cannot provide (e.g., asking for 9 GPUs when only 8 exist).

---

## 6. Team Workflow & Code Submission

- **No Manual Tinkering:** Do not SSH into the head node to manually edit running scripts.
- **Pull Requests are Mandatory:** All architectural changes, resource adjustments (e.g., changing `num_gpus`), and environment dependency updates must be submitted via GitHub Pull Requests.
- **Task Tracking:** Centralize implementation details and error logs in GitHub Issues so the entire backend team has a searchable history of how cluster bottlenecks were resolved.
- **Graceful Shutdowns:** Always ensure your driver scripts finish cleanly or call `ray.shutdown()` at the end to release resources back to the pool.

---

## 7. Ray Tasks

A **Task** is a standard Python function decorated with `@ray.remote`. It is completely **stateless** — it takes inputs, performs a computation, returns an output, and immediately forgets everything.

Because Tasks hold no state, the Ray Scheduler can distribute thousands of them across all 8 nodes simultaneously.

**Best used for:** data processing, array manipulations, image resizing, independent simulations.

```python
import ray

@ray.remote
def add_numbers(a, b):
    return a + b

# Executes asynchronously somewhere on the cluster
future = add_numbers.remote(5, 5)
result = ray.get(future)  # Blocks until complete
```