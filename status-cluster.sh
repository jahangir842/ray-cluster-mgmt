#!/usr/bin/env bash
# Filename follows the workstation command requested for this project.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster-common.sh"
load_hosts
heading "STATUS"

host_args=()
for i in "${!host_users[@]}"; do
  host_args+=("${host_users[i]}" "${host_addresses[i]}" "${host_roles[i]}")
done
printf -v remote_args '%q ' "${host_args[@]}"
if run_step "Query Ray head · $head_address" \
  ssh "${ssh_options[@]}" "$head_user@$head_address" "bash -s -- $remote_args" <<'REMOTE'
set -euo pipefail
for ray_bin in "$HOME"/miniconda3/envs/*/bin/ray; do
  if [[ -x "$ray_bin" ]]; then
    export PATH="$(dirname "$ray_bin"):$PATH"
    break
  fi
done
timeout 20 python - "$@" <<'PY'
import sys
import ray

hosts = list(zip(*[iter(sys.argv[1:])] * 3))
head_address = next(ip for _, ip, role in hosts if role == "head")
try:
    ray.init(address="auto", logging_level="ERROR")
except ConnectionError:
    print(f"Ray head is stopped or unavailable on {head_address}.")
    raise SystemExit(1)
try:
    nodes = [node for node in ray.nodes() if node["Alive"]]
    total = ray.cluster_resources()
    available = ray.available_resources()
    print()
    print(f"  {'PC':<5} {'ADDRESS':<17} {'ROLE':<8} {'STATE':<12} {'CPU':>5} {'GPU':>5} {'RAM GiB':>9} {'STORE GiB':>11}")
    print("  " + "─" * 88)
    healthy = True
    for user, ip, role in hosts:
        matches = [node for node in nodes if node["NodeManagerAddress"] == ip]
        state = "ONLINE" if len(matches) == 1 else "MISSING" if not matches else "DUPLICATE"
        healthy = healthy and len(matches) == 1
        resources = {}
        for node in matches:
            for key, value in node["Resources"].items():
                resources[key] = resources.get(key, 0) + value
        print(f"  {user:<5} {ip:<17} {role:<8} {state:<12} {resources.get('CPU', 0):>5g} {resources.get('GPU', 0):>5g} {resources.get('memory', 0) / 2**30:>9.1f} {resources.get('object_store_memory', 0) / 2**30:>11.1f}")
    print()
    print(f"  Active Ray nodes: {len(nodes)}  |  Expected: {len(hosts)}")
    for key, label, divisor in (("CPU", "CPU", 1), ("GPU", "GPU", 1), ("memory", "RAM GiB", 2**30), ("object_store_memory", "Object store GiB", 2**30)):
        capacity = total.get(key, 0) / divisor
        free = available.get(key, 0) / divisor
        print(f"  {label:<18} {max(0, capacity - free):>7.1f} used / {capacity:g} total")
    print("\n  RAM is Ray scheduling capacity, not GPU VRAM or physical RAM usage.")
finally:
    ray.shutdown()
# Report incomplete clusters without discarding the table.
if not healthy or len(nodes) != len(hosts):
    print("  WARNING: expected exactly one Ray node per configured computer.")
print(f"\n  Dashboard: http://{head_address}:8265\n")
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
