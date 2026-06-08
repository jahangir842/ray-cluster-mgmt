# Ray Cluster Management

Welcome to the **Ray Cluster Management**! This repository contains everything you need to deploy a Ray cluster from scratch—from local development to production-grade Kubernetes deployments. Clone this repo to any machine and follow the step-by-step guides to get a fully functional Ray cluster running.

---

## Table of Contents

1. [Introduction to Ray](#introduction-to-ray)
2. [Deployment Methods](#deployment-methods)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [Prerequisites](#prerequisites)

---

## Introduction to Ray

[**Ray**](https://ray.io/) is an open-source, distributed computing framework that makes it simple to scale Python and AI workloads from a single machine to a cluster of thousands of nodes. Whether you're running machine learning training, hyperparameter tuning, or distributed data processing, Ray handles the complex aspects of distributed computing—task scheduling, object storage, fault tolerance, and communication—transparently.

### The Ray Ecosystem

Ray provides a suite of **high-level libraries** built on top of its core distributed execution engine:

- **Ray Core:** Low-level distributed task and actor API.
- **Ray Train:** Distributed training for PyTorch, TensorFlow, XGBoost, and HuggingFace models.
- **Ray Tune:** Hyperparameter optimization and population-based training.
- **Ray Serve:** Model serving and online inference with autoscaling.
- **Ray Data:** Distributed data loading and preprocessing at scale.
- **Ray AIR (AI Runtime):** An end-to-end ML platform combining Train, Tune, and Serve.

---

## Deployment Methods

This workshop covers **three production-ready deployment strategies**, each suited for different environments:

| **Method** | **Environment** | **Complexity** | **Best For** | **Detailed Guide** |
|:--|:--|:--|:--|:--|
| **Manual CLI** | Systemwise | ⭐ Low | Learning, local testing, small fixed clusters | [01-manual-cli](installation/01-manual-cli/README.md) |
| **Docker Compose** | Single Host (Multi-container) | ⭐⭐ Medium | Local development, isolated testing, demo environments | [02-docker-compose](installation/02-docker-compose/README.md) |
| **KubeRay** | Kubernetes Cluster | ⭐⭐⭐ High | Production, auto-scaling, cloud-native environments | [03-kuberay](installation/03-kuberay/README.md) |
| **MLflow** | Head Node (shared storage) | ⭐ Low | Experiment tracking and artifact storage for all nodes | [04-mlflow](installation/04-mlflow/README.md) |
| **Shared Storage** | All Nodes (NFS) | ⭐ Low | `/mnt/cluster_storage` mounted on every node for checkpoints and artifacts | [05-shared-storage](installation/05-shared-storage/README.md) |

### Quick Comparison Table

| Feature | Manual CLI | Docker Compose | KubeRay |
|:--|:--|:--|:--|
| **Dependency Isolation** | ❌ System-level | ✅ Container-level | ✅ Container + Orchestration |
| **Multi-node Simulation** | ❌ Requires separate machines | ✅ On single host (via networking) | ✅ Multi-node clusters |
| **Auto-scaling** | Manual | Manual (via `docker-compose scale`) | ✅ Automatic (Kubernetes-native) |
| **Ease of Setup** | 5 minutes | 10 minutes | 30+ minutes (requires K8s cluster) |
| **Production Ready** | ✅ Yes (for fixed clusters) | ⚠️ Mostly (for single-host) | ✅✅ Yes (full production) |

---

## Repository Structure

```
ray-cluster-mgmt/
├── README.md                           # This file
├── developers-guide.md                 # Developer notes and workflow
├── requirements.txt                    # Python dependencies
├── mlflow-command.txt                  # MLflow server reference commands
├── installation/                       # Cluster deployment methods
│   ├── 01-manual-cli/                  # Manual SSH-based multi-node deployment
│   ├── 02-docker-compose/              # Multi-node simulation with Docker
│   ├── 03-kuberay/                     # Production deployment on Kubernetes
│   ├── 04-mlflow/                      # MLflow tracking server setup
│   │   ├── README.md
│   │   └── start_mlflow_server.sh      # MLflow server startup script
│   └── 05-shared-storage/              # NFS shared storage setup for all nodes
│       └── README.md
└── examples/                           # Example workloads by Ray library
    ├── 1-ray-core/                     # Raw @ray.remote tasks (stateless parallelism)
    │   ├── matrix-multiplcation/
    │   └── squre-root/
    ├── 2-ray-training/                 # Distributed PyTorch training with Ray Train
    ├── 3-RL-lib/                       # Reinforcement learning with Ray RLlib
    ├── 4-ray-serve(LLM)/               # LLM serving with Ray Serve
    ├── 5-ray-tune/                     # Hyperparameter search with Ray Tune
    └── 6.vllm/                         # LLM serving with vLLM and Transformers
        ├── 1.vllm-with-ray-backend.md  # Pipeline-parallel across 8 GPU nodes via Ray
        ├── 2.vllm-on-single-node.md    # Single-GPU vLLM serving
        └── 3.serve-with-transformers.md # Serving with HuggingFace Transformers
```

---

## Examples

The `examples/` directory contains ready-to-run workloads organized by Ray library:

| # | Folder | What it covers |
|:--|:--|:--|
| 1 | `1-ray-core/` | Stateless parallelism with `@ray.remote` — matrix multiplication, square root across the cluster |
| 2 | `2-ray-training/` | Distributed PyTorch training with `ray.train.torch.TorchTrainer`; checkpoints to `/mnt/cluster_storage/` |
| 3 | `3-RL-lib/` | Reinforcement learning experiments with Ray RLlib (PPO) |
| 4 | `4-ray-serve(LLM)/` | LLM serving with native Ray Serve (no vLLM) |
| 5 | `5-ray-tune/` | Hyperparameter search using Ray Tune with Optuna backend |
| 6 | `6.vllm/` | LLM serving with vLLM (single-node and 8-GPU multi-node via Ray) and HuggingFace Transformers |

### Example 6 — LLM Serving

Three guides in [`examples/6.vllm/`](examples/6.vllm/):

| Guide | Description |
|-------|-------------|
| [`1.vllm-with-ray-backend.md`](examples/6.vllm/1.vllm-with-ray-backend.md) | vLLM pipeline-parallel serving across **8 GPUs on 8 nodes** using Ray as the distributed executor |
| [`2.vllm-on-single-node.md`](examples/6.vllm/2.vllm-on-single-node.md) | vLLM single-GPU serving on one node (no Ray required) |
| [`3.serve-with-transformers.md`](examples/6.vllm/3.serve-with-transformers.md) | Quick inference and FastAPI server using HuggingFace Transformers directly |

---

## Quick Start

Choose your deployment method and follow the direct commands (no scripts—execute step by step to learn!).

---

## Common Commands by Method

### Manual CLI Commands

```bash
ray start --head              # Start head node
ray status                    # Check cluster status
ray stop                      # Stop Ray cluster
ray start --address='IP:6379' # Connect worker to head
```

### Docker Compose Commands

```bash
docker-compose up -d                  # Start cluster
docker-compose ps                     # List containers
docker-compose logs -f                # View logs
docker-compose up -d --scale ray-worker=5  # Scale workers
docker-compose down                   # Stop cluster
```

### KubeRay Commands

```bash
kubectl apply -f ray-cluster.yaml           # Deploy cluster
kubectl get raycluster                      # List clusters
kubectl describe raycluster my-ray-cluster  # Details
kubectl delete raycluster my-ray-cluster    # Delete cluster
kubectl logs <pod-name>                     # View pod logs
```

---

## Prerequisites

### System Requirements
- **OS:** Linux (Ubuntu 20.04+), macOS, or Windows (WSL2)
- **Disk Space:** 5 GB minimum
- **RAM:** 4 GB minimum (8 GB recommended for multi-node testing)

### Required Software (varies by method)

#### For Method 1 (Manual CLI):
- Python 3.8+
- pip
- Basic networking tools (`ssh`, `curl`)

#### For Method 2 (Docker Compose):
- Docker 20.10+
- Docker Compose 1.29+

#### For Method 3 (KubeRay):
- Kubernetes cluster (kind, minikube, or cloud K8s)
- `kubectl` 1.20+
- Helm 3+ (optional, for KubeRay operator)

---

## Next Steps

1. **Choose your deployment method** based on your environment and needs.
2. **Navigate to the corresponding folder** (`01-*`, `02-*`, or `03-*`).
3. **Follow the detailed `README.md`** in that folder.
4. **Run the example jobs** to verify your cluster is working.
5. **Adapt the setup** to your own workloads.

---

## Resources

- [Ray Official Documentation](https://docs.ray.io/)
- [Ray GitHub Repository](https://github.com/ray-project/ray)
- [Ray Cluster Launcher Documentation](https://docs.ray.io/en/latest/cluster/getting-started.html)
- [KubeRay Project](https://github.com/ray-project/kuberay)
- [Ray Community Slack](https://docs.ray.io/en/latest/community/index.html)

---

## License

This workshop repository is provided as-is for educational purposes. Please refer to individual Ray projects for their respective licenses.

---

Happy clustering! 🚀
