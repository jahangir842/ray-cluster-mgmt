#!/usr/bin/env bash
# Run from the workstation; locate Ray in a Miniconda environment on each node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster-common.sh"
load_hosts
heading "START"

if ! run_step "$head_user · $head_address · head ready" ssh "${ssh_options[@]}" "$head_user@$head_address" "bash -s -- $head_address" <<'REMOTE'
set -euo pipefail
for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
  if [[ -x "$ray_bin" ]]; then
    export PATH="$(dirname "$ray_bin"):$PATH"
    break
  fi
done
export RAY_USAGE_STATS_ENABLED=0
if timeout 15 ray status --address="$1:6379" >/dev/null 2>&1; then
  echo "Ray head is already running."
else
  ray start --head --node-ip-address="$1" --port=6379 \
    --dashboard-host=0.0.0.0 --dashboard-port=8265
fi
REMOTE
then
  printf '\n  Head unavailable; workers cannot join a cluster without it.\n' >&2
  exit 1
fi

if ! run_step "Read connected nodes" ssh "${ssh_options[@]}" "$head_user@$head_address" 'bash -s' <<'REMOTE'
set -euo pipefail
for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
  if [[ -x "$ray_bin" ]]; then
    export PATH="$(dirname "$ray_bin"):$PATH"
    break
  fi
done
timeout 20 python - <<'PY'
import ray
ray.init(address="auto", logging_level="ERROR")
try:
    print(" ".join(node["NodeManagerAddress"] for node in ray.nodes() if node["Alive"]))
finally:
    ray.shutdown()
PY
REMOTE
then
  printf '\n  Could not query the head; retry startup once it is available.\n' >&2
  exit 1
fi
active_nodes="$STEP_OUTPUT"

result=0
for i in "${!host_users[@]}"; do
  [[ "${host_roles[i]}" == worker ]] || continue
  user="${host_users[i]}" address="${host_addresses[i]}"
  if [[ " $active_nodes " == *" $address "* ]]; then
    notice "$user · $address · already connected"
    continue
  fi
  if ! run_step "$user · $address · worker ready" ssh "${ssh_options[@]}" "$user@$address" "bash -s -- $address $head_address" <<'REMOTE'
set -euo pipefail
for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
  if [[ -x "$ray_bin" ]]; then
    export PATH="$(dirname "$ray_bin"):$PATH"
    break
  fi
done
export RAY_USAGE_STATS_ENABLED=0
ray start --address="$2:6379" --node-ip-address="$1"
REMOTE
  then
    result=1
  fi
done

if ! bash "$SCRIPT_DIR/status-cluster.sh"; then
  result=1
fi
exit "$result"
