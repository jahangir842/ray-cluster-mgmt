# Ray Cluster Management

Learning setup: a Ray cluster on the same four GPU computers used by
`dask-mgmt`, managed from the workstation using the scripts below or by
running the Ray CLI directly in one SSH terminal per computer. Each computer
has 32 logical CPUs and an RTX 4500 Ada 24 GB GPU.

| Computer | Address | SSH access | Ray role |
|---|---|---|---|
| `pc4` | `192.168.3.74` | `ssh pc4@192.168.3.74` | Head + worker |
| `pc5` | `192.168.3.75` | `ssh pc5@192.168.3.75` | Worker |
| `pc6` | `192.168.3.76` | `ssh pc6@192.168.3.76` | Worker |
| `pc7` | `192.168.3.77` | `ssh pc7@192.168.3.77` | Worker |
| Workstation | `192.168.3.81` | Local terminal | Manages the cluster and copies the demo to pc4 |

All four cluster computers already have Miniconda in `~/miniconda3` and use
the existing **`dask-env`** with Python 3.12. Ray 2.59.0 is installed there.
No Ray installation is needed on the workstation for this workflow.

## Scripts to Start, Stop, and Check the Whole Cluster

Run these from a **local workstation terminal**, inside this repository.
They SSH into pc4–pc7 and use `~/miniconda3/envs/dask-env` directly, so no
local conda activation is needed. SSH key access must already work for all
four computers; the scripts use non-interactive SSH. The node usernames,
addresses, and roles are configured in [`hosts.txt`](hosts.txt).

```bash
cd /home/jahangir/projects/ray-cluster-mgmt
```

**Start the whole cluster:**

```bash
./start-cluster.sh
```

The [start script](start-cluster.sh) starts the head on pc4, then the
workers on pc5–pc7. Each node advertises **4 CPUs, 1 GPU, 8 GiB of RAM
scheduling capacity, and a 1 GiB object store** (16 CPUs and 4 GPUs total).
The manual CPU-only demo below uses no GPUs.
It skips an existing head and workers already connected to it, then prints
a table of node states, CPU/GPU counts, RAM scheduling capacity, and object
store capacity, plus the dashboard address.

**Stop the whole cluster:**

```bash
./stop-cluster.sh
```

The [stop script](stop-cluster.sh) stops workers first, then pc4.
It attempts all four computers even if one fails, and returns a nonzero exit
code if any SSH or stop command fails. Both scripts leave Dask services running.

**Check cluster status:**

```bash
./status-cluster.sh
```

The [status script](status-cluster.sh) shows each computer by name and IP,
flags missing or duplicate Ray nodes, and summarizes used/total Ray resources.
RAM figures are Ray scheduling capacity, not GPU VRAM or measured physical
RAM usage.

All three scripts show colored headings and concise progress messages in a
terminal, with plain output when redirected or when `NO_COLOR` is set.
Successful Ray command logs are hidden. On failure, they show a short error
excerpt and the path to the complete log under `/tmp/ray-cluster.*` on the
workstation. The scripts use the shared `cluster-common.sh` beside them.

## Quick Start (Manual SSH, CPU Only)

This demo uses **4 CPUs per computer, 16 CPUs total, and no GPUs**. Dask's
GPU workers can remain running. Ray CPU resources control task scheduling;
they do not reserve physical CPUs exclusively.

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

In **each SSH terminal**, activate the environment before running Ray commands:

```bash
conda activate dask-env
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
  --dashboard-host=0.0.0.0 --dashboard-port=8265 \
  --num-cpus=4 --num-gpus=0 --object-store-memory=1073741824 \
  --memory=8589934592
```

This starts the head and allows pc4 to run tasks too; no second Ray process
is needed on pc4. The command returns with Ray running in the background.
If Ray is already running, skip starting it again or follow the stop step first.

Dashboard: <http://192.168.3.74:8265>.

### 4. Start the workers

**pc5 terminal:**

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --address=192.168.3.74:6379 \
  --node-ip-address=192.168.3.75 --num-cpus=4 --num-gpus=0 \
  --object-store-memory=1073741824 --memory=8589934592
```

**pc6 terminal:**

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --address=192.168.3.74:6379 \
  --node-ip-address=192.168.3.76 --num-cpus=4 --num-gpus=0 \
  --object-store-memory=1073741824 --memory=8589934592
```

**pc7 terminal:**

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --address=192.168.3.74:6379 \
  --node-ip-address=192.168.3.77 --num-cpus=4 --num-gpus=0 \
  --object-store-memory=1073741824 --memory=8589934592
```

Each computer contributes a 1 GiB object store and advertises 8 GiB of memory
for scheduling. These commands also run Ray in the background, so closing
an SSH terminal does not stop the cluster.

**pc4 terminal**, check the cluster:

```bash
ray status
```

Expect **4 active nodes**, **16 CPUs**, and **4 GiB total object store memory**.

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
Use `ray stop` on all four. This leaves the Dask services running.

## More Examples

The first demo is [`examples/1-ray-core/cluster-demo.py`](examples/1-ray-core/cluster-demo.py).
Other workloads live under `examples/`: Ray Core tasks, distributed training,
RLlib, Ray Serve, Ray Tune, and LLM serving. Those workloads may need additional
dependencies, GPU resources, or shared storage beyond this CPU demo.

The [minimal demo guide](docs/minimal-ray-demo.md) also provides workstation
SSH commands for submission and shutdown. Older guides may reference the
previous head at `192.168.3.73`; this setup uses **`192.168.3.74`**.
