#!/usr/bin/env bash
# Filename follows the workstation command requested for this project.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster-common.sh"
heading "STATUS"

if run_step "Query Ray head · 192.168.3.74" \
  ssh "${ssh_options[@]}" pc4@192.168.3.74 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH="$HOME/miniconda3/envs/dask-env/bin:$PATH"
timeout 20 python - <<'PY'
import ray

try:
    ray.init(address="auto", logging_level="ERROR")
except ConnectionError:
    print("Ray head is stopped or unavailable on pc4.")
    raise SystemExit(1)
try:
    nodes = [node for node in ray.nodes() if node["Alive"]]
    total = ray.cluster_resources()
    available = ray.available_resources()
    print()
    print(f"  {'PC':<5} {'ADDRESS':<17} {'ROLE':<8} {'STATE':<12} {'CPU':>5} {'GPU':>5} {'RAM GiB':>9} {'STORE GiB':>11}")
    print("  " + "─" * 88)
    healthy = True
    for n in (4, 5, 6, 7):
        ip = f"192.168.3.7{n}"
        matches = [node for node in nodes if node["NodeManagerAddress"] == ip]
        state = "ONLINE" if len(matches) == 1 else "MISSING" if not matches else "DUPLICATE"
        healthy = healthy and len(matches) == 1
        resources = {}
        for node in matches:
            for key, value in node["Resources"].items():
                resources[key] = resources.get(key, 0) + value
        print(f"  {'pc' + str(n):<5} {ip:<17} {'head' if n == 4 else 'worker':<8} {state:<12} {resources.get('CPU', 0):>5g} {resources.get('GPU', 0):>5g} {resources.get('memory', 0) / 2**30:>9.1f} {resources.get('object_store_memory', 0) / 2**30:>11.1f}")
    print()
    print(f"  Active Ray nodes: {len(nodes)}  |  Expected: 4")
    for key, label, divisor in (("CPU", "CPU", 1), ("GPU", "GPU", 1), ("memory", "RAM GiB", 2**30), ("object_store_memory", "Object store GiB", 2**30)):
        capacity = total.get(key, 0) / divisor
        free = available.get(key, 0) / divisor
        print(f"  {label:<18} {max(0, capacity - free):>7.1f} used / {capacity:g} total")
    print("\n  RAM is Ray scheduling capacity, not GPU VRAM or physical RAM usage.")
    print("  Dashboard: http://192.168.3.74:8265\n")
finally:
    ray.shutdown()
# Report incomplete clusters without discarding the table.
if not healthy or len(nodes) != 4:
    print("  WARNING: expected exactly one Ray node per computer.")
PY
REMOTE
then
  printf '%s\n' "$STEP_OUTPUT"
  if [[ "$STEP_OUTPUT" == *"WARNING:"* ]]; then
    exit 1
  fi
else
  printf '\n  %sCluster status unavailable or incomplete.%s\n' "$red" "$reset" >&2
  printf '  If the head is stopped, worker state cannot be determined from it.\n' >&2
  printf '  Start: ./start-cluster.sh  |  Stop: ./stop-cluster.sh\n\n' >&2
  exit 1
fi
