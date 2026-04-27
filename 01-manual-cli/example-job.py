#!/usr/bin/env python3
"""
Simple Ray Example Job

This script demonstrates basic Ray functionality:
- Initializing Ray
- Creating and running remote tasks
- Processing results
"""

import ray
import time
from datetime import datetime

# Initialize Ray
# If Ray was started with `ray start --head`, this will connect to it
# Otherwise, it will start a local Ray instance
ray.init(ignore_reinit_error=True)

print(f"[{datetime.now()}] Ray initialized successfully!")
print(f"[{datetime.now()}] Ray cluster info:")
print(f"  - Nodes: {len(ray.nodes())}")
print(f"  - Available CPUs: {ray.available_resources().get('CPU', 0)}")
print(f"  - Available GPUs: {ray.available_resources().get('GPU', 0)}")
print()

# Define a remote function
@ray.remote
def expensive_computation(x):
    """A simple computation that takes a bit of time"""
    import time
    time.sleep(1)  # Simulate work
    return x * x

# Define another remote function
@ray.remote
def process_list(items):
    """Process a list of items"""
    return [item * 2 for item in items]

# Example 1: Running multiple tasks in parallel
print("[Task 1] Running 10 parallel computations...")
start_time = time.time()

# Submit tasks
futures = [expensive_computation.remote(i) for i in range(10)]

# Get results (blocks until all complete)
results = ray.get(futures)
elapsed = time.time() - start_time

print(f"  Results: {results}")
print(f"  Time taken: {elapsed:.2f}s (sequential would take ~10s)")
print()

# Example 2: Chained operations
print("[Task 2] Running chained operations...")
future1 = expensive_computation.remote(5)
future2 = process_list.remote([1, 2, 3, 4, 5])

result1 = ray.get(future1)
result2 = ray.get(future2)

print(f"  Task 1 result: {result1}")
print(f"  Task 2 result: {result2}")
print()

# Example 3: Using Ray Actors (stateful computation)
print("[Task 3] Using Ray Actors (stateful classes)...")

@ray.remote
class Counter:
    """A stateful actor that maintains a counter"""
    def __init__(self, initial_value=0):
        self.value = initial_value
    
    def increment(self):
        self.value += 1
        return self.value
    
    def get_value(self):
        return self.value

# Create an actor
counter = Counter.remote(initial_value=0)

# Increment multiple times
for i in range(5):
    result = ray.get(counter.increment.remote())
    print(f"  Counter value: {result}")

final_value = ray.get(counter.get_value.remote())
print(f"  Final counter value: {final_value}")
print()

# Cleanup
print(f"[{datetime.now()}] Shutting down Ray...")
ray.shutdown()

print(f"[{datetime.now()}] Done!")
