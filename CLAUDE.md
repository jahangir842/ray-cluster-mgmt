# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Ray Cluster Deployment Workshop** — a learning and reference repository for deploying and using Ray distributed computing clusters. It covers three deployment strategies (manual CLI, Docker Compose, KubeRay) and a growing library of example workloads.

The physical cluster used in development: **8 nodes, 268 CPUs, 8 GPUs**, head node at `192.168.3.73`.

---

## Cluster Connection

All scripts that run on or against the cluster must initialize Ray with `address='auto'` (when running from within a cluster node):

```python
ray.init(address='auto')
```

For distributed training jobs, always set the NCCL/GLOO interface env vars at init time to avoid inter-node communication failures:

```python
ray.init(
    address="auto",
    runtime_env={
        "env_vars": {
            "NCCL_SOCKET_IFNAME": "enp0s31f6,eno1",
            "GLOO_SOCKET_IFNAME": "enp0s31f6,eno1",
        }
    }
)
```

For remote connections from a laptop: `ray.init(address="ray://192.168.3.73:10001")`

---

## Job Submission

**Recommended — Ray Job CLI** (enables full dashboard observability):
```bash
ray job submit \
    --address="http://192.168.3.73:8265" \
    --working-dir . \
    -- python my_script.py
```

**Quick testing only — direct execution:**
```bash
python my_script.py
```

**Useful job management commands:**
```bash
ray job list                          # List all jobs
ray job logs <job_id> --follow        # Stream live logs
ray job stop <job_id>                 # Cancel a job
ray status                            # Check cluster resources
```

**Dashboard:** `http://192.168.3.73:8265` (requires internal network or VPN)

---

## Resource Allocation Rules

Always declare resources explicitly on `@ray.remote` — Ray defaults to 1 CPU / 0 GPU if omitted, which can monopolize a node:

```python
@ray.remote(num_gpus=1, num_cpus=4)
def gpu_task(data): ...

# Fractional GPU sharing (4 tasks per GPU)
@ray.remote(num_gpus=0.25)
def small_inference_task(data): ...
```

For large datasets shared across many tasks, use `ray.put()` to store once in the object store rather than passing the data directly into each task call.

---

## Deployment Methods

| Method | Folder | Use Case |
|--------|--------|----------|
| Manual CLI | `01-manual-cli/` | Bare-metal multi-node, learning |
| Docker Compose | `02-docker-compose/` | Local dev, single-host multi-container |
| KubeRay | `03-kuberay/` | Production, Kubernetes, auto-scaling |

### Docker Compose
```bash
cd 02-docker-compose
docker-compose build
docker-compose up -d
docker-compose exec ray-head python /app/example-job.py
docker-compose down
```

### KubeRay
```bash
cd 03-kuberay
helm install kuberay-operator kuberay/kuberay-operator --namespace kuberay-system --create-namespace
kubectl apply -f ray-cluster.yaml
kubectl port-forward svc/my-ray-cluster-head-svc 8265:8265 &
kubectl apply -f sample-job.yaml
```

---

## Examples Architecture

`examples/` is organized by Ray library:

- **`1-ray-core/`** — Raw `@ray.remote` tasks (stateless parallelism). Scripts show single-machine vs. cluster comparison for the same workload.
- **`2-ray-training/`** — `ray.train.torch.TorchTrainer` for distributed PyTorch. Uses `prepare_model()` + `prepare_data_loader()` pattern. Checkpoints go to `/mnt/cluster_storage/`. Profiler scripts write TensorBoard traces and per-rank memory timelines there too.
- **`3-RL-lib/`** — `ray.rllib` PPO experiments using `PPOConfig().build_algo()`.
- **`4-ray-serve(LLM)/`** — LLM serving (no vLLM; use native Ray Serve).
- **`5-ray-tune/`** — Hyperparameter search (prefer Optuna backend).

### Ray Train V2

Enable V2 API for all new training scripts:
```python
import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
```

`ScalingConfig` pattern for distributed training:
```python
scaling_config = ray.train.ScalingConfig(num_workers=7, use_gpu=True)
trainer = ray.train.torch.TorchTrainer(train_func, scaling_config=scaling_config)
result = trainer.fit()
```

For large models (8B+), use FSDP via `torch.distributed.fsdp.fully_shard` with a device mesh; load models on `"meta"` device first, then `.to_empty(device=device)` to avoid OOM during initialization.

---

## Dependencies

Install for training examples:
```bash
pip install ray[default] torch torchvision transformers accelerate huggingface_hub tensorboard sentencepiece protobuf matplotlib
```

Or: `pip install -r requirements.txt` (covers the transformer/profiling stack; does not include `ray` or `torch` themselves).

---

## Team Workflow

- Submit all changes via Pull Requests — no direct edits to running scripts on the head node.
- Always call `ray.shutdown()` at the end of driver scripts.
- Track architectural decisions and cluster bottleneck resolutions in GitHub Issues.
