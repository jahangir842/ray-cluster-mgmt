# Ray Cluster: Developer Guidelines

Welcome to the Ray Cluster. This 8-node environment provides 268 CPUs, 8 GPUs, and massive shared memory specifically designed for heavy, distributed Python workloads. 

To ensure the cluster remains stable and resource contention is minimized, all developers must adhere to the following guidelines when writing and submitting Ray jobs.

## 1. Connecting to the Cluster

Do not run your heavy Python execution loops directly on the head node's OS. Instead, your Python scripts should connect to the existing cluster.

**The Standard Connection:**
At the top of your Python driver script, initialize the connection. If you are running the script from within a node on the cluster, use `'auto'`:

```python
import ray

# Connect to the existing cluster
ray.init(address='auto')

# Verify connection
print(ray.cluster_resources())
```

*Note: If you are connecting remotely from your local laptop (using Ray Client), you will use the `ray://<head-node-ip>:10001` format.*

## 2. Resource Allocation (The Golden Rule)

Ray uses a logical resource accounting system. **You must explicitly declare the resources your tasks need.** If you do not, Ray defaults to 1 CPU per task and 0 GPUs, which can quickly overwhelm a single node or leave your GPUs sitting idle.

### Allocating CPUs and GPUs
Use the `@ray.remote` decorator to define exactly what your function requires.

```python
# GOOD: Explicitly requesting 1 GPU and 4 CPUs
@ray.remote(num_gpus=1, num_cpus=4)
def process_heavy_tensor(data):
    # Your GPU-intensive code here
    pass

# BAD: No resources defined. Ray assumes 1 CPU, 0 GPUs.
@ray.remote
def process_heavy_tensor_bad(data):
    pass
```

### Fractional GPUs
If Junaid, Qasim, or Hammad are working on smaller inference models that do not require an entire GPU, you can slice the GPU logically to allow multiple actors to share it.

```python
# Allows 4 of these actors to run concurrently on a single GPU
@ray.remote(num_gpus=0.25)
class SmallInferenceModel:
    pass
```

## 3. Memory Management & The Object Store

Our cluster relies heavily on the `object_store_memory` (Zero-Copy shared memory) for speed. 

* **Avoid Anti-Patterns:** Do not pass massive arrays (like 10GB datasets) as direct arguments to multiple tasks inside a loop. This creates duplicate copies in standard RAM and will crash the node.
* **Use `ray.put()`:** If multiple tasks need to read the same massive dataset, place it in the object store *once*, and pass the lightweight reference to your tasks.

```python
# 1. Put the massive dataset into the shared Object Store
massive_data_ref = ray.put(load_massive_dataset())

# 2. Pass the reference (ObjectRef) to the workers, NOT the actual data
results = [worker_task.remote(massive_data_ref, i) for i in range(10)]
```

## 4. Observability & Dashboard Access

While your tasks are running, you do not need to fly blind. The Ray Dashboard provides a real-time visual interface to monitor your specific workloads, hardware utilization, and application logs without needing SSH access.

**Accessing the Dashboard:**
Navigate to `http://<HEAD_NODE_IP>:8265` in your web browser. *(Note: You must be on the internal network or connected to the office VPN to access this UI).*

**What Developers Should Monitor:**
* **Jobs View:** Track the execution of your specific Python driver scripts. You can see if your job is actively running, pending resources, or failed.
* **Actors View:** Check the health of your stateful classes (e.g., your inference models). If an actor crashes due to an Out-Of-Memory (OOM) error, the traceback will be flagged here.
* **Logs View:** This is the most powerful feature for debugging. You can click on any worker or actor to view its standard output (`stdout`) and error logs (`stderr`) directly in the browser, eliminating the need to hunt down log files across 8 physical machines.
* **Metrics & Demands:** Watch the memory and GPU utilization spike when your tasks hit the cluster. If your task is sitting in the "Pending Demands" state indefinitely, it means you requested resources that the cluster physically cannot provide (e.g., asking for 9 GPUs).

## 5. Team Workflow & Code Submission

To maintain architectural consistency and prevent rogue jobs from causing cluster out-of-memory (OOM) errors, all Ray-dependent code must follow our standard deployment pipeline:

* **No Manual Tinkering:** Do not SSH into the head node to manually edit running scripts.
* **Pull Requests are Mandatory:** All architectural changes, resource adjustments (e.g., changing `num_gpus`), and environment dependency updates must be submitted via **GitHub Pull Requests**.
* **Task Tracking:** If you are optimizing a Ray Task or debugging a memory leak, centralize the implementation details and error logs in **GitHub Issues** rather than direct messages. This ensures the entire backend team has a searchable history of how we solved cluster bottlenecks.
* **Graceful Shutdowns:** Always ensure your driver scripts finish cleanly or call `ray.shutdown()` at the end of execution to release resources back to the pool.
