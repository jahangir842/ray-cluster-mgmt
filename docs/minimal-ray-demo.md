# Minimal four-node Ray demo

pc4 (`192.168.3.74`) is the Ray head and also runs tasks. Ray automatically
detects all available CPUs and GPUs: 32 logical CPUs and 1 GPU per node,
128 logical CPUs and 4 GPUs total. Memory and object store sizes use Ray defaults.
The demo uses CPU tasks; the cluster also supports GPU workloads.

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
    for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
      if [[ -x "$ray_bin" ]]; then
        export PATH="$(dirname "$ray_bin"):$PATH"
        break
      fi
    done
    python -m pip freeze > /tmp/ray-demo-before.txt
    python -m pip install \
      -c /tmp/ray-demo-before.txt "ray[default]==2.59.0"
    python -m pip check
  '
    done
```

## Start

From the workstation, use `./start-cluster.sh`, `./status-cluster.sh`, and
`./stop-cluster.sh` to manage the PCs listed in the repository's `hosts.txt`.
Each entry is `SSH_USER IP_ADDRESS ROLE`, where the role is `head` or `worker`.
Exactly one head is required; blank lines and `#` comments are allowed.
These scripts use automatic CPU, GPU, memory, and object store detection.
Restart existing Ray processes to apply resource changes.

Startup skips already connected workers and continues after failed worker
connections, then displays status and the dashboard URL. The head must be
available before workers can join. Stop attempts every worker before the head,
even if some PCs are unreachable. Commands return a nonzero exit status if a
node fails or the cluster is incomplete. They manage Ray processes on powered-on
PCs; they do not turn PCs on or off.

For manual startup, select the first Miniconda environment containing Ray in
each SSH terminal (or activate your preferred Ray environment):

```bash
    for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
      if [[ -x "$ray_bin" ]]; then
        export PATH="$(dirname "$ray_bin"):$PATH"
        break
      fi
    done
```

On **pc4**:

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --head \
  --node-ip-address=192.168.3.74 --port=6379 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265
```

On **each of pc5, pc6, pc7**:

```bash
RAY_USAGE_STATS_ENABLED=0 ray start --address=192.168.3.74:6379
```

These commands return after starting background processes; SSH terminals can
close. On pc4, run `ray status`: expect 4 active nodes, 128 logical CPUs, and 4 GPUs.

## Run

From this repository on the workstation, copy the demo to pc4 and submit it:

```bash
ssh pc4@192.168.3.74 'mkdir -p ~/ray-demo'
scp examples/1-ray-core/cluster-demo.py pc4@192.168.3.74:ray-demo/
ssh pc4@192.168.3.74 '
    for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
      if [[ -x "$ray_bin" ]]; then
        export PATH="$(dirname "$ray_bin"):$PATH"
        break
      fi
    done
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
  ssh pc$n@192.168.3.7$n 'bash -s' <<'REMOTE'
for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
  if [[ -x "$ray_bin" ]]; then
    export PATH="$(dirname "$ray_bin"):$PATH"
    break
  fi
done
ray stop
REMOTE
done
```

This stops Ray processes on all four computers.

References: [Ray installation](https://docs.ray.io/en/latest/ray-overview/installation.html),
[manual on-premises startup](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html).
