# Ray Cluster — Docker Compose (IaC)

Parameterized Docker Compose setup for the physical Ray cluster. One compose
file runs on every node; a **profile** picks the role:

| Profile | Runs on | Role |
|---------|---------|------|
| `head` | 192.168.3.73 | Ray head (GCS, dashboard, client server) |
| `worker` | each worker node | Ray worker (CPU + GPU) |

All tunables live in `.env` — no editing YAML files.

---

## File Structure

```
02-docker-compose/
├── .env.example                # Config template (committed)
├── .env                        # Your values (git-ignored)
├── Dockerfile                  # Parameterized Ray image (CPU or GPU via BASE_IMAGE)
├── docker-compose.yml          # head/worker by profile, host networking
└── scripts/
    └── deploy-cluster.sh       # SSH-based deploy to all nodes

# The smoke-test job lives at repo root: examples/example-job.py
# (mounted into the head container at /app/examples/)
```

---

## Quick Start

### Option A — Manual (per node)

**On the head node (192.168.3.73):**
```bash
cd installation/02-docker-compose
cp .env.example .env                     # set RAY_HEAD_HOST, HEAD_NUM_CPUS, etc.
docker compose --profile head up -d
docker compose --profile head exec ray-head ray status
```

**On each worker node:**
```bash
# copy the .env, Dockerfile, and docker-compose.yml to the worker, then:
cp .env.example .env                     # set RAY_HEAD_HOST, WORKER_NUM_CPUS, WORKER_NUM_GPUS
docker compose --profile worker up -d
```

Open the dashboard at **http://192.168.3.73:8265**, and run the test job with:
```bash
docker compose --profile head exec ray-head python /app/examples/example-job.py
```

### Option B — Automated SSH deploy

```bash
# On the head node, with SSH access to all workers:
cp .env.example .env
# set WORKER_HOSTS="192.168.3.74 192.168.3.75 ..."
# set SSH_USER=ubuntu (or your user)

bash scripts/deploy-cluster.sh up     # builds + deploys head, then SSHes into each worker
bash scripts/deploy-cluster.sh down   # tear down everything
```

The script rsyncs `docker-compose.yml`, `Dockerfile`, and `.env` to
`/opt/ray-cluster/` on each worker before starting containers with the
`worker` profile.

---

## Choosing the Right Image

The `BASE_IMAGE` variable in `.env` controls what gets built:

```bash
# CPU-only nodes
BASE_IMAGE=rayproject/ray:2.47.0-py311

# GPU nodes (CUDA + all GPU drivers pre-installed)
BASE_IMAGE=rayproject/ray:2.47.0-py311-gpu
```

The head defaults to the CPU image and workers default to the GPU image. To
override at build time on a specific node:

```bash
BASE_IMAGE=rayproject/ray:2.47.0-py311-gpu \
  docker compose --profile worker up -d --build
```

---

## Command Reference

`docker-compose.yml` is the default file, so no `-f` flag is needed — just pick
the profile for the node's role.

```bash
# ── Head node ────────────────────────────────────────────────
docker compose --profile head up -d                              # start head
docker compose --profile head down                               # stop head
docker compose --profile head build                              # (re)build the image
docker compose --profile head logs -f                            # follow head logs
docker compose --profile head exec ray-head ray status           # cluster resources
docker compose --profile head exec ray-head bash                 # shell in the head container
docker compose --profile head exec ray-head python /app/examples/example-job.py   # test job

# ── Worker node ──────────────────────────────────────────────
docker compose --profile worker up -d                            # start worker
docker compose --profile worker down                             # stop worker

# ── Whole cluster over SSH (from the head node) ──────────────
bash scripts/deploy-cluster.sh up
bash scripts/deploy-cluster.sh down
```

---

## Key .env Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_IMAGE` | `rayproject/ray:2.47.0-py311` | Docker base image |
| `RAY_HEAD_HOST` | `192.168.3.73` | Head node IP |
| `RAY_GCS_PORT` | `6379` | Ray GCS port |
| `RAY_DASHBOARD_PORT` | `8265` | Dashboard port |
| `RAY_CLIENT_PORT` | `10001` | Ray client (remote `ray://`) port |
| `HEAD_NUM_CPUS` | `4` | CPUs reserved on head |
| `HEAD_OBJECT_STORE_MEMORY` | `4000000000` | Head object store (bytes) |
| `WORKER_NUM_CPUS` | `32` | CPUs per worker |
| `WORKER_NUM_GPUS` | `1` | GPUs per worker |
| `WORKER_OBJECT_STORE_MEMORY` | `8000000000` | Worker object store (bytes) |
| `SHARED_STORAGE_PATH` | `/mnt/cluster_storage` | Path to shared NFS storage |
| `NCCL_SOCKET_IFNAME` | `enp0s31f6,eno1` | NIC for NCCL (multi-GPU) |
| `GLOO_SOCKET_IFNAME` | `enp0s31f6,eno1` | NIC for Gloo (multi-GPU) |
| `WORKER_HOSTS` | *(empty)* | Space-separated worker IPs for SSH deploy |
| `SSH_USER` | `ubuntu` | SSH user for worker deploy |

---

## Networking

Every node uses `network_mode: host` so Ray processes bind to the real host IP
without NAT. This is required for inter-node Ray communication and NCCL GPU
collectives. Workers reach the head at `RAY_HEAD_HOST:RAY_GCS_PORT`
(`192.168.3.73:6379` by default).

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
docker compose --profile head logs -f    # look for "Dashboard available at" line
```
