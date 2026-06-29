#!/usr/bin/env bash
# Stop the BARE-METAL (conda) Ray cluster on every node so the Docker cluster can
# take over port 6379 / the GPUs. RUN THIS ON THE HEAD NODE.
#
# This frees the cluster for the Docker bring-up. It does NOT touch any conda env;
# it only stops the running raylets. Re-joining bare-metal later is just `ray start`.
set -uo pipefail

NODES=(192.168.3.71 192.168.3.72 192.168.3.73 192.168.3.74 192.168.3.75 192.168.3.76 192.168.3.77 192.168.3.78)
SSH_USER="user"

# Per-node path to the `ray` binary. Most nodes use anaconda3; .72 uses .conda.
ray_bin_for() {
  case "$1" in
    192.168.3.72) echo '$HOME/.conda/envs/ray-env/bin/ray' ;;
    *)            echo '$HOME/anaconda3/envs/ray-env/bin/ray' ;;
  esac
}

for ip in "${NODES[@]}"; do
  rb="$(ray_bin_for "$ip")"
  echo ">> ray stop on $ip"
  ssh -o BatchMode=yes "$SSH_USER@$ip" "$rb stop 2>/dev/null | tail -1 || true"
done

echo ">> bare-metal Ray stopped on all nodes."
