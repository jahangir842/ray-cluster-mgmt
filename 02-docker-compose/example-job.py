#!/usr/bin/env python3
"""
Ray Example Job for Docker Containers

This script runs on the Ray cluster and demonstrates:
- Connecting to the Ray cluster
- Running distributed tasks
- Processing results
"""

import ray
import time
from datetime import datetime

print(f"[{datetime.now()}] Starting Ray job...")

# Initialize Ray
# When running in Docker Compose, connect to the head node by hostname
try:
    ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)
except:
    # Fallback for local Ray instance
    ray.init(ignore_reinit_error=True)

print(f"[{datetime.now()}] Ray initialized successfully!")

# Get cluster information
cluster_resources = ray.available_resources()
print(f"[{datetime.now()}] Cluster resources:")
print(f"  - CPUs: {cluster_resources.get('CPU', 0)}")
print(f"  - GPUs: {cluster_resources.get('GPU', 0)}")
print(f"  - Memory: {cluster_resources.get('memory', 0) / 1e9:.2f} GB")

node_info = ray.nodes()
print(f"  - Nodes: {len(node_info)}")

# Define a remote function
@ray.remote
def cpu_bound_task(task_id, duration=2):
    """A task that does some CPU-bound work"""
    import time
    import os
    hostname = os.environ.get('HOSTNAME', 'unknown')
    
    print(f"    Task {task_id} starting on {hostname}")
    time.sleep(duration)
    result = sum(i * i for i in range(1000000))
    print(f"    Task {task_id} finished on {hostname}")
    
    return {"task_id": task_id, "result": result, "hostname": hostname}

# Run distributed tasks
print(f"[{datetime.now()}] Submitting 6 tasks to the cluster...")
start_time = time.time()

# Submit tasks
futures = [cpu_bound_task.remote(i) for i in range(6)]

# Wait for results
print(f"[{datetime.now()}] Waiting for results...")
results = ray.get(futures)

elapsed = time.time() - start_time

print(f"[{datetime.now()}] Results received in {elapsed:.2f}s:")
for result in results:
    print(f"  - Task {result['task_id']:02d}: result={result['result']}, host={result['hostname']}")

# Shutdown Ray
print(f"[{datetime.now()}] Shutting down Ray...")
ray.shutdown()

print(f"[{datetime.now()}] Job complete!")
