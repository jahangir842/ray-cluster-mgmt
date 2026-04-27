# Ray Cluster Deployment Workshop

Welcome to the **Ray Cluster Deployment Workshop**! This repository contains everything you need to deploy a Ray cluster from scratch—from local development to production-grade Kubernetes deployments. Clone this repo to any machine and follow the step-by-step guides to get a fully functional Ray cluster running.

---

## Table of Contents

1. [Introduction to Ray](#introduction-to-ray)
2. [Core Uses of Ray](#core-uses-of-ray)
3. [Why Deploy Ray?](#why-deploy-ray)
4. [Deployment Methods](#deployment-methods)
5. [Repository Structure](#repository-structure)
6. [Quick Start](#quick-start)
7. [Prerequisites](#prerequisites)

---

## Introduction to Ray

[**Ray**](https://ray.io/) is an open-source, distributed computing framework that makes it simple to scale Python and AI workloads from a single machine to a cluster of thousands of nodes. Whether you're running machine learning training, hyperparameter tuning, or distributed data processing, Ray handles the complex aspects of distributed computing—task scheduling, object storage, fault tolerance, and communication—transparently.

### Key Characteristics

- **Unified API:** Write code once, then run it locally or on a cluster without modification.
- **Low Latency:** Microsecond-level task dispatch and sub-millisecond object transfers via in-memory storage (the Ray object store).
- **Fault Tolerant:** Automatic recovery from node failures with lineage-based recovery.
- **Language Agnostic:** Native support for Python; additional support for Java, C++, and other languages via extensions.
- **Multi-Tenant Ready:** Share a single Ray cluster across multiple teams/applications with resource isolation.

### The Ray Ecosystem

Ray provides a suite of **high-level libraries** built on top of its core distributed execution engine:

- **Ray Core:** Low-level distributed task and actor API.
- **Ray Train:** Distributed training for PyTorch, TensorFlow, XGBoost, and HuggingFace models.
- **Ray Tune:** Hyperparameter optimization and population-based training.
- **Ray Serve:** Model serving and online inference with autoscaling.
- **Ray Data:** Distributed data loading and preprocessing at scale.
- **Ray AIR (AI Runtime):** An end-to-end ML platform combining Train, Tune, and Serve.

---

## Core Uses of Ray

Ray excels in scenarios where you need to parallelize workloads and scale beyond a single machine. Here are the primary use cases:

### 1. **Distributed Machine Learning Training**
   - Scale PyTorch, TensorFlow, JAX, or HuggingFace models across multiple GPUs and nodes.
   - **Example:** Train a large language model on 100 GPUs using Ray Train with automatic distributed data parallelism.
   - **Benefit:** Reduce training time from weeks to hours without rewriting your training code.

### 2. **Hyperparameter Tuning & AutoML**
   - Run hundreds or thousands of model configurations in parallel.
   - **Example:** Test 500 parameter combinations simultaneously across your cluster using Ray Tune.
   - **Benefit:** Find optimal model configurations 10-100x faster than sequential tuning.

### 3. **Model Serving & Inference**
   - Deploy multiple models with auto-scaling endpoints via Ray Serve.
   - **Example:** Serve a recommendation engine that scales from 10 to 10,000 RPS automatically.
   - **Benefit:** Reduce latency and cost by provisioning exactly the resources you need, when you need them.

### 4. **Batch Inference & Prediction**
   - Process millions of data points through a trained model in parallel.
   - **Example:** Run inference on 1 billion images using a GPU cluster without memory overflow.
   - **Benefit:** Complete batch predictions in minutes instead of hours/days.

### 5. **Large-Scale Data Processing**
   - Transform, shuffle, and aggregate terabytes of data without exceeding node memory.
   - **Example:** Preprocess a 10TB dataset for training ML models on a 8-node cluster.
   - **Benefit:** Avoid moving data to slow external systems; process in-cluster with Ray Data.

### 6. **Distributed Simulation & RL**
   - Parallelize thousands of environment rollouts for reinforcement learning.
   - **Example:** Train a game-playing AI by running 1000s of simulations in parallel.
   - **Benefit:** Significantly faster training for RL agents through massive rollout parallelism.

### 7. **Real-Time Analytics & ETL**
   - Stream data through distributed Python pipelines with backpressure handling.
   - **Example:** Process streaming logs, detect anomalies, and trigger alerts in real-time across nodes.
   - **Benefit:** Handle high-throughput data pipelines without building custom distributed systems.

---

## Why Deploy Ray?

### Scaling Without Complexity
With Ray, you write your code once and scale it effortlessly:

```python
# Your laptop: works on a sample of data
result = process_data(small_dataset)

# Your cluster: same code, but scales to millions of rows
result = process_data(large_dataset)
```

### Heterogeneous Workloads
Ray supports mixed workloads: some tasks use CPUs, others GPUs, memory-intensive operations coexist safely:

```python
@ray.remote(num_cpus=2)
def cpu_intensive_task():
    return compute_something()

@ray.remote(num_gpus=1)
def gpu_intensive_task():
    return train_model()

results = ray.get([
    cpu_intensive_task.remote(),
    gpu_intensive_task.remote()
])
```

### Cost Efficiency
- Run only when needed and auto-scale down during idle periods.
- Share a single cluster across multiple teams/projects.
- Use spot instances (cloud) or on-premise hardware efficiently.

---

## Deployment Methods

This workshop covers **three production-ready deployment strategies**, each suited for different environments:

| **Method** | **Environment** | **Complexity** | **Best For** | **Folder** |
|:--|:--|:--|:--|:--|
| **Manual CLI** | Bare-metal / Single Host | ⭐ Low | Learning, local testing, small fixed clusters | `01-manual-cli` |
| **Docker Compose** | Single Host (Multi-container) | ⭐⭐ Medium | Local development, isolated testing, demo environments | `02-docker-compose` |
| **KubeRay** | Kubernetes Cluster | ⭐⭐⭐ High | Production, auto-scaling, cloud-native environments | `03-kuberay` |

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
├── 01-manual-cli/                      # Manual deployment using `ray start`
│   ├── README.md                       # Setup instructions
│   ├── setup-environment.sh            # Automated environment setup
│   └── example-job.py                  # Sample Ray job
├── 02-docker-compose/                  # Multi-node simulation with Docker
│   ├── README.md                       # Docker setup instructions
│   ├── docker-compose.yml              # Multi-container cluster definition
│   ├── Dockerfile                      # Ray environment image
│   └── example-job.py                  # Sample Ray job for containers
└── 03-kuberay/                         # Production deployment on Kubernetes
    ├── README.md                       # KubeRay setup instructions
    ├── ray-cluster.yaml                # Ray cluster CRD
    ├── ray-autoscaler.yaml             # Autoscaler configuration
    └── sample-job.yaml                 # Example Kubernetes Job
```

---

## Quick Start

Choose your deployment method and follow the direct commands (no scripts—execute step by step to learn!).

### Option 1️⃣: Manual CLI Setup (5-10 minutes)

**Best for:** Learning, local testing, prototyping  
**Requirements:** Python 3.8+  
**Complexity:** ⭐ Easiest

Execute these commands directly:

```bash
# Step 1: Create isolated Python environment
python3 -m venv ray-env
source ray-env/bin/activate

# Step 2: Install Ray framework
pip install --upgrade pip
pip install ray[default]

# Step 3: Start Ray (keep this terminal open)
ray start --head

# Step 4: In a NEW terminal, run example job
python 01-manual-cli/example-job.py

# Step 5: View dashboard in browser
# Open: http://localhost:8265

# Step 6: Stop Ray when done
ray stop
```

**What you'll learn:**
- Setting up isolated Python environments
- Ray startup process
- Submitting distributed tasks
- Using the Ray Dashboard for monitoring

See [01-manual-cli/README.md](01-manual-cli/README.md) for detailed step-by-step guide.

---

### Option 2️⃣: Docker Compose Setup (10 minutes)

**Best for:** Local development, demos, testing without Python setup  
**Requirements:** Docker 20.10+, Docker Compose 1.29+  
**Complexity:** ⭐⭐ Medium

```bash
# Step 1: Navigate to Docker folder
cd 02-docker-compose

# Step 2: Build the Docker image
docker-compose build

# Step 3: Start the cluster (1 head + 2 workers)
docker-compose up -d

# Step 4: Check status
docker-compose logs -f ray-head

# Step 5: Run example job
docker-compose exec ray-head python /app/example-job.py

# Step 6: View dashboard
# Open: http://localhost:8265

# Step 7: Stop the cluster
docker-compose down
```

**What you'll learn:**
- Containerizing Ray clusters
- Multi-node simulation on single host
- Docker networking and container management

See [02-docker-compose/README.md](02-docker-compose/README.md) for detailed guide and monitoring.

---

### Option 3️⃣: Kubernetes (KubeRay) Deployment (30+ minutes)

**Best for:** Production, cloud deployments, auto-scaling  
**Requirements:** Kubernetes cluster 1.24+  
**Complexity:** ⭐⭐⭐ Advanced

```bash
# Step 1: Navigate to KubeRay folder
cd 03-kuberay

# Step 2: Install KubeRay operator (one-time setup)
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator \
  --namespace kuberay-system --create-namespace

# Step 3: Deploy Ray cluster
kubectl apply -f ray-cluster.yaml

# Step 4: Wait for cluster to be ready
kubectl get raycluster -w

# Step 5: Port-forward dashboard
kubectl port-forward svc/my-ray-cluster-head-svc 8265:8265 &

# Step 6: View dashboard
# Open: http://localhost:8265

# Step 7: Submit example job
kubectl apply -f sample-job.yaml
kubectl logs -f job/my-ray-job

# Step 8: Clean up
kubectl delete raycluster my-ray-cluster
```

**What you'll learn:**
- Kubernetes custom resources (CRDs)
- deploying with KubeRay operator
- Auto-scaling in production
- Cloud-native distributed systems

See [03-kuberay/README.md](03-kuberay/README.md) for detailed guide and configuration.

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