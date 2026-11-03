#!/usr/bin/env bash
# Run from the workstation; Ray is installed in dask-env on every node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster-common.sh"
heading "START"

run_step "pc4 · 192.168.3.74 · head ready" ssh "${ssh_options[@]}" pc4@192.168.3.74 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH="$HOME/miniconda3/envs/dask-env/bin:$PATH"
export RAY_USAGE_STATS_ENABLED=0
if timeout 15 ray status --address=192.168.3.74:6379 >/dev/null 2>&1; then
  echo "Ray head is already running."
else
  ray start --head --node-ip-address=192.168.3.74 --port=6379 \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --num-cpus=4 --num-gpus=0 --object-store-memory=1073741824 \
    --memory=8589934592
fi
REMOTE

run_step "Read connected nodes" ssh "${ssh_options[@]}" pc4@192.168.3.74 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH="$HOME/miniconda3/envs/dask-env/bin:$PATH"
python - <<'PY'
import ray
ray.init(address="auto", logging_level="ERROR")
try:
    print(" ".join(node["NodeManagerAddress"] for node in ray.nodes() if node["Alive"]))
finally:
    ray.shutdown()
PY
REMOTE
active_nodes="$STEP_OUTPUT"

for node in 5 6 7; do
  address="192.168.3.7$node"
  if [[ " $active_nodes " == *" $address "* ]]; then
    notice "pc$node · $address · already connected"
    continue
  fi
  run_step "pc$node · $address · worker ready" ssh "${ssh_options[@]}" "pc$node@$address" "bash -s -- $address" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/miniconda3/envs/dask-env/bin:$PATH"
export RAY_USAGE_STATS_ENABLED=0
ray start --address=192.168.3.74:6379 --node-ip-address="$1" \
  --num-cpus=4 --num-gpus=0 --object-store-memory=1073741824 \
  --memory=8589934592
REMOTE
done

exec bash "$SCRIPT_DIR/staus-cluster.sh"
