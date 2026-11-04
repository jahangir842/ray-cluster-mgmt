# Minimal four-node Ray demo

Use the existing `dask-env` on pc4–pc7. pc4 (`192.168.3.74`) is the Ray
head and also runs tasks. This CPU demo advertises 4 CPUs and 0 GPUs per
node, with a 1 GiB object store per node. Dask's GPU workers can remain running.
Ray CPU resources are scheduling reservations, not OS CPU isolation.

Verified on 2026-11-03: Ray 2.59.0 installed on all four nodes, `pip check`
passed everywhere, all 52 pre-existing package pins per node were preserved,
and the demo succeeded through the Ray Jobs CLI on all four computers:

```text
Connected to 4 nodes
192.168.3.74 (hdwc0147): 333332833333500000
192.168.3.75 (HDWC0148): 2333331833333500000
192.168.3.76 (hdwc0146): 6333330833333500000
192.168.3.77 (hdwc0145): 12333329833333500000
PASS: sum of squares for 4,000,000 integers = 21333325333334000000
```

## Install

From the workstation, install the same Ray version on every node, constraining
existing packages to their installed versions:

```bash
for n in 4 5 6 7; do
  ssh pc$n@192.168.3.7$n '
    set -e
    ~/miniconda3/envs/dask-env/bin/python -m pip freeze > /tmp/ray-demo-before.txt
    ~/miniconda3/envs/dask-env/bin/python -m pip install \
      -c /tmp/ray-demo-before.txt "ray[default]==2.59.0"
    ~/miniconda3/envs/dask-env/bin/python -m pip check
  '
done
```

## Start

From the workstation, use `./start-cluster.sh`, `./status-cluster.sh`, and
`./stop-cluster.sh` to manage the PCs listed in the repository's `hosts.txt`.
Each entry is `SSH_USER IP_ADDRESS ROLE`, where the role is `head` or `worker`.
Exactly one head is required; blank lines and `#` comments are allowed.
These scripts advertise 4 CPUs, 1 GPU, 8 GiB of scheduling RAM, and a 1 GiB
object store per node. Restart existing Ray processes to apply resource changes.

Startup skips already connected workers and continues after failed worker
connections, then displays status and the dashboard URL. The head must be
available before workers can join. Stop attempts every worker before the head,
even if some PCs are unreachable. Commands return a nonzero exit status if a
node fails or the cluster is incomplete. They manage Ray processes on powered-on
PCs; they do not turn PCs on or off.

The manual commands below retain the original CPU-only demo configuration.

On **pc4**, in a terminal with `conda activate dask-env`:

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --head \
  --node-ip-address=192.168.3.74 --port=6379 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 \
  --num-cpus=4 --num-gpus=0 --object-store-memory=1073741824 \
  --memory=8589934592
```

On **each of pc5, pc6, pc7**, with `conda activate dask-env`:

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --address=192.168.3.74:6379 \
  --num-cpus=4 --num-gpus=0 --object-store-memory=1073741824 \
  --memory=8589934592
```

These commands return after starting background processes; SSH terminals can
close. On pc4, run `ray status`: expect 4 active nodes and 16 CPUs.

## Run

From this repository on the workstation, copy the demo to pc4 and submit it:

```bash
ssh pc4@192.168.3.74 'mkdir -p ~/ray-demo'
scp examples/1-ray-core/cluster-demo.py pc4@192.168.3.74:ray-demo/
ssh pc4@192.168.3.74 '
  export PATH="$HOME/miniconda3/envs/dask-env/bin:$PATH"
  ray job submit --address=http://192.168.3.74:8265 \
    --working-dir "$HOME/ray-demo" -- python cluster-demo.py
'
```

The demo sends one sum-of-squares task to each live node and prints each
computer's IP, hostname, and subtotal. It checks the combined result against
the exact formula and prints `PASS`. This verifies distributed execution;
the small workload is not a speed benchmark.

Dashboard: <http://192.168.3.74:8265>. No Ray installation is needed on the
workstation because submission runs over SSH on pc4.

## Stop Ray

From the workstation:

```bash
for n in 5 6 7 4; do
  ssh pc$n@192.168.3.7$n '~/miniconda3/envs/dask-env/bin/ray stop'
done
```

This stops Ray processes and leaves the Dask services running.

References: [Ray installation](https://docs.ray.io/en/latest/ray-overview/installation.html),
[manual on-premises startup](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html).
