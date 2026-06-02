# MLflow Tracking Server Setup

This guide covers deploying an MLflow tracking server on the Ray head node, backed by shared cluster storage so all worker nodes can log experiments and access artifacts.

---

## Overview

| Setting | Value |
|:--|:--|
| **Host** | `0.0.0.0` (all interfaces) |
| **Port** | `5000` |
| **Backend store** | `/mnt/cluster_storage/mlflow/mlflow.db` (SQLite) |
| **Artifact root** | `/mnt/cluster_storage/mlflow/artifacts` |
| **UI / Tracking URI** | `http://192.168.3.73:5000` |

---

## Prerequisites

- Shared storage mounted at `/mnt/cluster_storage/` on all nodes.
- Ray environment active (`conda activate ray-env`).
- MLflow installed: `pip install --upgrade mlflow`

---

## Option 1: Automated Script (Recommended)

Run the startup script from the `installation/04-mlflow/` directory on the head node:

```bash
cd installation/04-mlflow
chmod +x start_mlflow_server.sh
./start_mlflow_server.sh
```

The script will:
1. Verify shared storage is mounted
2. Install/upgrade MLflow
3. Create required directories
4. Stop any existing MLflow process on port 5000
5. Launch the server in the background
6. Wait up to 30 seconds for the server to become ready
7. Print the tracking URI and management commands

---

## Option 2: Manual Launch via SSH

From any machine with SSH access to the head node:

```bash
ssh user@192.168.3.73 "
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ray-env
nohup mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri /mnt/cluster_storage/mlflow \
  --default-artifact-root /mnt/cluster_storage/mlflow/artifacts \
  --serve-artifacts \
  --cors-allowed-origins '*' \
  --allowed-hosts '*' \
  > /tmp/mlflow.log 2>&1 &
echo 'MLflow started, PID:' \$!
"
```

---

## Connecting Training Scripts

Set the tracking URI in your Ray training scripts before logging:

```python
import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.3.73:5000"
mlflow.set_tracking_uri("http://192.168.3.73:5000")
```

Or export it in your shell before submitting jobs:

```bash
export MLFLOW_TRACKING_URI=http://192.168.3.73:5000
ray job submit --address="http://192.168.3.73:8265" --working-dir . -- python my_training_script.py
```

---

## Management Commands

```bash
# View server logs
tail -f /tmp/mlflow_server.log

# Stop the server (script-managed)
kill $(cat /tmp/mlflow_server.pid)

# Stop the server (manual launch)
kill $(lsof -ti:5000)

# Check if server is running
curl -s http://192.168.3.73:5000/health
```

---

## Directory Layout

```
/mnt/cluster_storage/mlflow/
├── mlflow.db           # SQLite experiment/run metadata
└── artifacts/          # Logged model artifacts, plots, files
```

---

## Access

| Resource | URL |
|:--|:--|
| **MLflow UI** | `http://192.168.3.73:5000` |
| **REST API** | `http://192.168.3.73:5000/api/2.0/mlflow/...` |
| **Health check** | `http://192.168.3.73:5000/health` |

> Requires internal network or VPN access to `192.168.3.73`.
