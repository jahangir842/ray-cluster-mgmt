#!/usr/bin/env bash
# Seed the vLLM image onto ALL nodes the cheap way: pull ONCE on the head, then
# distribute over the LAN via an NFS tarball (no per-node internet pulls; sidesteps
# the docker-group + IPv6 issues). RUN ON THE HEAD. Logs to the tar's directory.
set -uo pipefail

IMAGE="vllm/vllm-openai:v0.6.6"
DIR="/mnt/cluster_storage/vllm-ray"
TAR="$DIR/vllm-openai-v0.6.6.tar"
WORKERS=(192.168.3.71 192.168.3.72 192.168.3.74 192.168.3.75 192.168.3.76 192.168.3.77 192.168.3.78)
SSH_USER="user"

mkdir -p "$DIR"

# These nodes intermittently resolve the registry to an unreachable IPv6 address.
# docker pull resumes cached layers, so retry until one attempt connects over IPv4.
if sudo docker images --format '{{.Repository}}:{{.Tag}}' | grep -qx "$IMAGE"; then
  echo "[seed] $(date +%T) image already present on head, skipping pull"
else
  ok=0
  for try in $(seq 1 20); do
    echo "[seed] $(date +%T) pull attempt $try ..."
    if sudo docker pull "$IMAGE"; then ok=1; break; fi
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "[seed] head pull FAILED after retries"; exit 1; }
fi

echo "[seed] $(date +%T) saving image to $TAR ..."
sudo docker save "$IMAGE" -o "$TAR"
sudo chmod 644 "$TAR"
echo "[seed] tar size: $(du -h "$TAR" | cut -f1)"

echo "[seed] $(date +%T) loading on workers from NFS (parallel)..."
for n in "${WORKERS[@]}"; do
  ( ssh -o BatchMode=yes "$SSH_USER@$n" "sudo docker load -i $TAR" \
      && echo "[seed]   $n loaded" || echo "[seed]   $n LOAD FAILED" ) &
done
wait

echo "[seed] $(date +%T) verification:"
for n in 192.168.3.73 "${WORKERS[@]}"; do
  printf "[seed]   %s: " "$n"
  ssh -o BatchMode=yes "$SSH_USER@$n" \
    "sudo docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '$IMAGE' && echo PRESENT || echo MISSING" 2>/dev/null
done
echo "[seed] $(date +%T) done."
