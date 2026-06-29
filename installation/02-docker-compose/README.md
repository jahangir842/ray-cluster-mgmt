# Ray Cluster — Docker Compose (IaC)

Parameterized Docker Compose setup for Ray clusters. Supports two modes:

| Mode | File | Use case |
|------|------|----------|
| **Single-host** | `docker-compose.yml` | Local dev, CI, workshops on one machine |
| **Multi-node head** | `docker-compose.head.yml` | Run on the physical head node (192.168.3.73) |
| **Multi-node worker** | `docker-compose.worker.yml` | Run on each physical worker node |

All tunables live in `.env` — no editing YAML files.

---

## File Structure

```
02-docker-compose/
├── .env.example                # Config template (committed)
├── .env                        # Your values (git-ignored)
├── Dockerfile                  # Parameterized Ray image (CPU or GPU via BASE_IMAGE)
├── docker-compose.yml          # Single-host: head + N workers
├── docker-compose.head.yml     # Multi-node: head only, host networking
├── docker-compose.worker.yml   # Multi-node: worker only, host networking + GPU
├── Makefile                    # Convenience targets (make help)
├── scripts/
│   └── deploy-cluster.sh       # SSH-based deploy to all 8 nodes
└── example-job.py              # Smoke-test job
```

---

## Quick Start (single-host)

```bash
cd installation/02-docker-compose

# 1. Configure
cp .env.example .env
# Edit .env: set HEAD_NUM_CPUS, WORKER_NUM_CPUS, WORKER_REPLICAS, etc.

# 2. Build image
make build

# 3. Start cluster (1 head + WORKER_REPLICAS workers)
make up

# 4. Verify
make status          # ray status from inside head container

# 5. Run a test job
make job

# 6. Open dashboard
#    http://localhost:8265

# 7. Scale workers without restart
make scale N=5

# 8. Tear down
make down
```

---

## Multi-node Cluster (8 physical nodes)

### Option A — Manual (per node)

**On the head node (192.168.3.73):**
```bash
cp .env.example .env   # set RAY_HEAD_HOST, HEAD_NUM_CPUS, etc.
make head-up
```

**On each worker node:**
```bash
# copy the .env, Dockerfile, and docker-compose.worker.yml to the worker
# then:
cp .env.example .env   # set RAY_HEAD_HOST, WORKER_NUM_CPUS, WORKER_NUM_GPUS
make worker-up
```

### Option B — Automated SSH deploy

```bash
# On the head node, with SSH access to all workers:
cp .env.example .env
# set WORKER_HOSTS="192.168.3.74 192.168.3.75 ..."
# set SSH_USER=ubuntu (or your user)

make cluster-up      # builds + deploys head, then SSHes into each worker
make cluster-status  # ray status showing all nodes
make cluster-down    # tear down everything
```

The script (`scripts/deploy-cluster.sh`) rsyncs the compose files + `.env` to
`/opt/ray-cluster/` on each worker before starting containers.

---

## Choosing the Right Image

The `BASE_IMAGE` variable in `.env` controls what gets built:

```bash
# CPU-only nodes (default)
BASE_IMAGE=rayproject/ray:2.47.0-py311

# GPU nodes (CUDA + all GPU drivers pre-installed)
BASE_IMAGE=rayproject/ray:2.47.0-py311-gpu
```

For a mixed cluster (CPU head + GPU workers), set different `BASE_IMAGE` values
in the head and worker `.env` files, or override at build time:

```bash
# On a GPU worker node
BASE_IMAGE=rayproject/ray:2.47.0-py311-gpu make worker-up
```

---

## All Make Targets

```
Single-host:
  make build            Build the Ray image
  make up               Start head + workers
  make down             Stop everything
  make scale N=5        Change worker count without restart
  make logs             Follow head container logs
  make status           ray status (cluster resources)
  make shell            bash inside the head container
  make job              Run example-job.py
  make clean            Remove containers, volumes, and built images

Multi-node:
  make head-up          Start head on this machine
  make head-down        Stop head on this machine
  make worker-up        Start worker on this machine
  make worker-down      Stop worker on this machine
  make cluster-up       SSH-deploy to all nodes
  make cluster-down     SSH-teardown all nodes
  make cluster-status   ray status from head
```

---

## Key .env Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_IMAGE` | `rayproject/ray:2.47.0-py311` | Docker base image |
| `RAY_HEAD_HOST` | `192.168.3.73` | Head node IP (multi-node) |
| `RAY_GCS_PORT` | `6379` | Ray GCS port |
| `RAY_DASHBOARD_PORT` | `8265` | Dashboard port |
| `RAY_CLIENT_PORT` | `10001` | Ray client (remote `ray://`) port |
| `HEAD_NUM_CPUS` | `4` | CPUs reserved on head |
| `HEAD_OBJECT_STORE_MEMORY` | `4000000000` | Head object store (bytes) |
| `WORKER_NUM_CPUS` | `32` | CPUs per worker |
| `WORKER_NUM_GPUS` | `1` | GPUs per worker (multi-node) |
| `WORKER_OBJECT_STORE_MEMORY` | `8000000000` | Worker object store (bytes) |
| `WORKER_REPLICAS` | `2` | Workers in single-host mode |
| `SHARED_STORAGE_PATH` | `/mnt/cluster_storage` | Path to shared NFS storage |
| `NCCL_SOCKET_IFNAME` | `enp0s31f6,eno1` | NIC for NCCL (multi-GPU) |
| `GLOO_SOCKET_IFNAME` | `enp0s31f6,eno1` | NIC for Gloo (multi-GPU) |
| `WORKER_HOSTS` | *(empty)* | Space-separated worker IPs for SSH deploy |
| `SSH_USER` | `ubuntu` | SSH user for worker deploy |

---

## Networking

**Single-host mode** uses a Docker bridge network (`ray-net`). Containers
reference each other by hostname (`ray-head:6379`).

**Multi-node mode** uses `network_mode: host` on every node so Ray processes
bind to the real host IP without NAT. This is required for inter-node Ray
communication and NCCL GPU collectives.

---

## Troubleshooting

**Workers can't reach the head:**
```bash
# From a worker node, verify the GCS port is reachable
nc -zv 192.168.3.73 6379
```

**GPU not visible inside the container:**
```bash
# Verify the NVIDIA container runtime is installed
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**NCCL hangs on multi-GPU training:**
- Check `NCCL_SOCKET_IFNAME` matches the actual interface name (`ip link`)
- Comma-separated list; do not use IP prefixes (`192.168.` is invalid here)
- See `memory/project_vllm_ray_gloo.md` for the Gloo equivalent

**Dashboard not loading:**
```bash
make logs    # look for "Dashboard available at" line
```
