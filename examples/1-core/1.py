"""
Matrix Multiplication Benchmark: Ray Cluster vs Single Machine (32 cores)
=========================================================================
Tests identical workloads across:
  - Single machine using 32 CPU cores (multiprocessing via numpy/concurrent.futures)
  - Ray cluster (distributed across all available nodes)

Usage:
  python matrix_benchmark.py [--matrix-size N] [--num-tasks T] [--output results.json]

Defaults:
  --matrix-size  2048    (2048x2048 float32 matrices per task)
  --num-tasks    64      (64 independent multiplications = ~total load)
  --output       benchmark_results.json
"""

import argparse
import time
import json
import os
import sys
import numpy as np
import ray
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ─────────────────────────────────────────────────────────────────────────────
# Shared task: one matrix multiplication
# ─────────────────────────────────────────────────────────────────────────────

def _single_matmul(args):
    """Worker for single-machine multiprocessing."""
    size, seed = args
    rng = np.random.default_rng(seed)
    A = rng.random((size, size), dtype=np.float32)
    B = rng.random((size, size), dtype=np.float32)
    C = np.matmul(A, B)
    return float(C.sum())          # return a scalar so pickling is cheap


@ray.remote
def _ray_matmul(size: int, seed: int) -> float:
    """Ray remote task: one matrix multiplication."""
    import numpy as np
    rng = np.random.default_rng(seed)
    A = rng.random((size, size), dtype=np.float32)
    B = rng.random((size, size), dtype=np.float32)
    C = np.matmul(A, B)
    return float(C.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────────

def run_single_machine(matrix_size: int, num_tasks: int, num_cores: int = 32):
    """Run all tasks on a single machine using ProcessPoolExecutor."""
    print(f"\n{'='*60}")
    print(f"  SINGLE MACHINE  ({num_cores} cores)")
    print(f"  Tasks: {num_tasks}  |  Matrix: {matrix_size}x{matrix_size}")
    print(f"{'='*60}")

    args = [(matrix_size, seed) for seed in range(num_tasks)]

    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(_single_matmul, a): i for i, a in enumerate(args)}
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % max(1, num_tasks // 10) == 0:
                pct = 100 * completed / num_tasks
                elapsed = time.perf_counter() - start
                print(f"  [{pct:5.1f}%]  {completed}/{num_tasks} tasks  |  {elapsed:.2f}s elapsed")

    wall_time = time.perf_counter() - start
    ops_per_sec = num_tasks / wall_time
    # Approx FLOPs: 2 * N^3 per matmul
    total_flops = num_tasks * 2 * (matrix_size ** 3)
    gflops = total_flops / wall_time / 1e9

    print(f"\n  ✓ Done in {wall_time:.3f}s")
    print(f"  Throughput : {ops_per_sec:.2f} tasks/s")
    print(f"  Performance: {gflops:.1f} GFLOP/s")

    return {
        "mode": "single_machine",
        "cores": num_cores,
        "matrix_size": matrix_size,
        "num_tasks": num_tasks,
        "wall_time_s": round(wall_time, 4),
        "tasks_per_second": round(ops_per_sec, 4),
        "gflops": round(gflops, 2),
    }


def run_ray_cluster(matrix_size: int, num_tasks: int):
    """Run all tasks distributed on the Ray cluster."""
    # Connect to existing cluster
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    cluster_resources = ray.cluster_resources()
    total_cpus = int(cluster_resources.get("CPU", 0))
    total_gpus = int(cluster_resources.get("GPU", 0))
    num_nodes = len([k for k in cluster_resources if k.startswith("node:")])

    print(f"\n{'='*60}")
    print(f"  RAY CLUSTER  ({total_cpus} CPUs, {total_gpus} GPUs, {num_nodes} nodes)")
    print(f"  Tasks: {num_tasks}  |  Matrix: {matrix_size}x{matrix_size}")
    print(f"{'='*60}")

    start = time.perf_counter()
    refs = [_ray_matmul.remote(matrix_size, seed) for seed in range(num_tasks)]

    results = []
    completed = 0
    check_interval = max(1, num_tasks // 10)

    while refs:
        done, refs = ray.wait(refs, num_returns=min(check_interval, len(refs)), timeout=300)
        batch = ray.get(done)
        results.extend(batch)
        completed += len(done)
        elapsed = time.perf_counter() - start
        pct = 100 * completed / num_tasks
        print(f"  [{pct:5.1f}%]  {completed}/{num_tasks} tasks  |  {elapsed:.2f}s elapsed")

    wall_time = time.perf_counter() - start
    ops_per_sec = num_tasks / wall_time
    total_flops = num_tasks * 2 * (matrix_size ** 3)
    gflops = total_flops / wall_time / 1e9

    print(f"\n  ✓ Done in {wall_time:.3f}s")
    print(f"  Throughput : {ops_per_sec:.2f} tasks/s")
    print(f"  Performance: {gflops:.1f} GFLOP/s")

    return {
        "mode": "ray_cluster",
        "total_cpus": total_cpus,
        "total_gpus": total_gpus,
        "num_nodes": num_nodes,
        "matrix_size": matrix_size,
        "num_tasks": num_tasks,
        "wall_time_s": round(wall_time, 4),
        "tasks_per_second": round(ops_per_sec, 4),
        "gflops": round(gflops, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(single: dict, cluster: dict):
    speedup = single["wall_time_s"] / cluster["wall_time_s"]
    throughput_gain = cluster["tasks_per_second"] / single["tasks_per_second"]

    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Matrix size : {single['matrix_size']}x{single['matrix_size']}  |  Tasks: {single['num_tasks']}")
    print()
    print(f"  {'':30s} {'Single':>10s}  {'Cluster':>10s}")
    print(f"  {'-'*52}")
    print(f"  {'Wall time (s)':30s} {single['wall_time_s']:>10.3f}  {cluster['wall_time_s']:>10.3f}")
    print(f"  {'Tasks/second':30s} {single['tasks_per_second']:>10.2f}  {cluster['tasks_per_second']:>10.2f}")
    print(f"  {'GFLOP/s':30s} {single['gflops']:>10.1f}  {cluster['gflops']:>10.1f}")
    print()
    print(f"  Speedup (cluster vs single) : {speedup:.2f}x")
    print(f"  Throughput gain             : {throughput_gain:.2f}x")
    if speedup >= 1:
        print(f"  → Cluster is {speedup:.2f}x FASTER")
    else:
        print(f"  → Single machine is {1/speedup:.2f}x faster (cluster overhead dominates)")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ray Cluster vs Single Machine Matrix Multiply Benchmark")
    parser.add_argument("--matrix-size", type=int, default=2048,
                        help="Matrix dimension N for NxN matrices (default: 2048)")
    parser.add_argument("--num-tasks",   type=int, default=64,
                        help="Number of independent matmul tasks (default: 64)")
    parser.add_argument("--num-cores",   type=int, default=32,
                        help="Cores to use for single-machine run (default: 32)")
    parser.add_argument("--output",      type=str, default="benchmark_results.json",
                        help="JSON file to save results (default: benchmark_results.json)")
    parser.add_argument("--skip-single", action="store_true",
                        help="Skip single-machine benchmark (cluster only)")
    parser.add_argument("--skip-cluster",action="store_true",
                        help="Skip cluster benchmark (single only)")
    args = parser.parse_args()

    available_cores = multiprocessing.cpu_count()
    num_cores = min(args.num_cores, available_cores)
    if num_cores < args.num_cores:
        print(f"[WARN] Requested {args.num_cores} cores but only {available_cores} available. Using {num_cores}.")

    print(f"\nMatrix Multiplication Benchmark")
    print(f"  Matrix size  : {args.matrix_size}x{args.matrix_size} (float32)")
    print(f"  Tasks        : {args.num_tasks}")
    print(f"  Single cores : {num_cores}")
    approx_mem_gb = (args.matrix_size ** 2 * 4 * 3) / 1e9   # 3 matrices, 4 bytes each
    print(f"  Mem/task est : {approx_mem_gb:.2f} GB  (A + B + C)")

    results = {}

    # ── Single machine ──────────────────────────────────────────────────────
    if not args.skip_single:
        single_result = run_single_machine(args.matrix_size, args.num_tasks, num_cores)
        results["single_machine"] = single_result
    else:
        print("\n[Skipping single-machine benchmark]")
        single_result = None

    # ── Ray cluster ─────────────────────────────────────────────────────────
    if not args.skip_cluster:
        cluster_result = run_ray_cluster(args.matrix_size, args.num_tasks)
        results["ray_cluster"] = cluster_result
    else:
        print("\n[Skipping Ray cluster benchmark]")
        cluster_result = None

    # ── Summary ─────────────────────────────────────────────────────────────
    if single_result and cluster_result:
        print_summary(single_result, cluster_result)
        results["speedup"] = round(single_result["wall_time_s"] / cluster_result["wall_time_s"], 4)

    # ── Save results ─────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()