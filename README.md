# Ray Cluster Management

Learning setup: a Ray cluster on four GPU computers, managed from the
workstation using the scripts below or by
running the Ray CLI directly in one SSH terminal per computer. Each computer
has 32 logical CPUs and an RTX 4500 Ada 24 GB GPU.

| Computer | Address | SSH access | Ray role |
|---|---|---|---|
| `pc4` | `192.168.3.74` | `ssh pc4@192.168.3.74` | Head + worker |
| `pc5` | `192.168.3.75` | `ssh pc5@192.168.3.75` | Worker |
| `pc6` | `192.168.3.76` | `ssh pc6@192.168.3.76` | Worker |
| `pc7` | `192.168.3.77` | `ssh pc7@192.168.3.77` | Worker |
| Workstation | `192.168.3.81` | Local terminal | Manages the cluster and copies the demo to pc4 |

All four cluster computers already have Miniconda in `~/miniconda3` and
Ray 2.59.0 installed in a Python 3.12 environment. The scripts select the first
Miniconda environment containing a Ray executable (in directory name order).
No Ray installation is needed on the workstation for this workflow.

## Manage the Cluster from Your Workstation

Run from this repository with SSH key access to the nodes in [`hosts.txt`](hosts.txt).
No local Ray installation or conda activation is needed.

```bash
./start-cluster.sh   # Start head and workers
./status-cluster.sh  # Show node status and resources
./stop-cluster.sh    # Stop workers, then head
```

Startup automatically detects all CPUs and GPUs: **32 logical CPUs and 1 GPU
per node, 128 logical CPUs and 4 GPUs total**. Ray chooses memory and object
store sizes automatically. To apply this configuration to an already running
cluster, run `./stop-cluster.sh` followed by `./start-cluster.sh`.

## Quick Start (Manual SSH)

The cluster exposes all available CPUs and GPUs. The small demo below uses CPU
tasks; GPU resources are available for other workloads. Ray resource counts
control scheduling; they do not reserve physical CPUs exclusively.

### 1. Open four SSH terminals

From the workstation, open a separate terminal for each computer and connect:

**Terminal 1 — pc4:**

```bash
ssh pc4@192.168.3.74
```

**Terminal 2 — pc5:**

```bash
ssh pc5@192.168.3.75
```

**Terminal 3 — pc6:**

```bash
ssh pc6@192.168.3.76
```

**Terminal 4 — pc7:**

```bash
ssh pc7@192.168.3.77
```

In **each SSH terminal**, select the installed Ray environment before running commands:

```bash
for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
  if [[ -x "$ray_bin" ]]; then
    export PATH="$(dirname "$ray_bin"):$PATH"
    break
  fi
done
```

### 2. Install once / check versions

Ray is already installed on all four computers. In **each SSH terminal**, check:

```bash
python -c "import sys, ray; print(sys.version.split()[0], ray.__version__)"
```

All four should report Python `3.12.16` and Ray `2.59.0`.
If installing again, run these commands directly on **each computer** to keep
existing package versions pinned:

```bash
python -m pip freeze > /tmp/ray-demo-before.txt
python -m pip install -c /tmp/ray-demo-before.txt 'ray[default]==2.59.0'
python -m pip check
```

### 3. Start the head on pc4

**pc4 terminal:**

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --head \
  --node-ip-address=192.168.3.74 --port=6379 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265
```

This starts the head and allows pc4 to run tasks too; no second Ray process
is needed on pc4. The command returns with Ray running in the background.
If Ray is already running, skip starting it again or follow the stop step first.

Dashboard: <http://192.168.3.74:8265>.

### 4. Start the workers

**pc5 terminal:**

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --address=192.168.3.74:6379 \
  --node-ip-address=192.168.3.75
```

**pc6 terminal:**

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --address=192.168.3.74:6379 \
  --node-ip-address=192.168.3.76
```

**pc7 terminal:**

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --address=192.168.3.74:6379 \
  --node-ip-address=192.168.3.77
```

Ray automatically detects CPU and GPU resources and sizes its memory and
object store from available memory. These commands run Ray in the background, so closing
an SSH terminal does not stop the cluster.

**pc4 terminal**, check the cluster:

```bash
ray status
```

Expect **4 active nodes**, **128 logical CPUs**, and **4 GPUs**. Memory and
object store totals depend on available memory when Ray starts.

### 5. Run the demo

**Workstation**, in a local terminal from this repository, copy the demo to pc4:

```bash
ssh pc4@192.168.3.74 'mkdir -p ~/ray-demo'
scp examples/1-ray-core/cluster-demo.py pc4@192.168.3.74:ray-demo/
```

**pc4 terminal**, submit the demo:

```bash
cd ~/ray-demo
ray job submit --address=http://192.168.3.74:8265 \
  --working-dir . -- python cluster-demo.py
```

The command streams the output and waits for the job to finish. The demo sends
one sum-of-squares calculation to each computer, combines the answers, and
checks the result against the exact formula.

Verified output from all four computers on 2026-11-03:

```text
Connected to 4 nodes
192.168.3.74 (hdwc0147): 333332833333500000
192.168.3.75 (HDWC0148): 2333331833333500000
192.168.3.76 (hdwc0146): 6333330833333500000
192.168.3.77 (hdwc0145): 12333329833333500000
PASS: sum of squares for 4,000,000 integers = 21333325333334000000
```

This small demo verifies distributed execution; it is not a speed benchmark.
The dashboard's **Jobs** page also shows the submitted job and its logs.

### 6. Stop the cluster

Run this directly in **each worker terminal (pc5, pc6, pc7)** first, then in
the **pc4 terminal**:

```bash
ray stop
```

Stopping Ray on pc4 alone does not stop the processes on the other computers.
Use `ray stop` on all four.

## More Examples

The first demo is [`examples/1-ray-core/cluster-demo.py`](examples/1-ray-core/cluster-demo.py).
Other workloads live under `examples/`: Ray Core tasks, distributed training,
RLlib, Ray Serve, Ray Tune, and LLM serving. Those workloads may need additional
dependencies, GPU resources, or shared storage beyond this CPU demo.

The [minimal demo guide](docs/minimal-ray-demo.md) also provides workstation
SSH commands for submission and shutdown. Older guides may reference the
previous head at `192.168.3.73`; this setup uses **`192.168.3.74`**.
