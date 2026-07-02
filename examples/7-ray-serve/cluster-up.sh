#!/usr/bin/env bash
# Bring up the same multi-node Ray cluster as ../6.vllm/docker-multinode, then
# serve DeepSeek-R1-70B via Ray Serve (serve_vllm.py) instead of the raw
# `vllm serve` CLI. RUN THIS ON THE HEAD NODE (192.168.3.73).
#
# Reuses node-up.sh from 6.vllm/docker-multinode so node provisioning (NIC
# detection, shm sizing, CUDA compat fix) stays a single source of truth.
#
# Prereq: no bare-metal or docker-multinode vLLM cluster already holding
# port 6379 / the GPUs. Run ../6.vllm/docker-multinode/stop-baremetal-ray.sh
# and/or ../6.vllm/docker-multinode/cluster-down.sh first if needed.
set -euo pipefail

HEAD_IP="192.168.3.73"
WORKERS=(192.168.3.71 192.168.3.72 192.168.3.74 192.168.3.75 192.168.3.76 192.168.3.77 192.168.3.78)
SSH_USER="user"
NAME="vllm-ray"
NFS_DIR="/mnt/cluster_storage/vllm-ray"
HERE="$(cd "$(dirname "$0")" && pwd)"
NODE_UP="$HERE/../6.vllm/docker-multinode/node-up.sh"

DOCKER="docker"; $DOCKER info >/dev/null 2>&1 || DOCKER="sudo docker"

echo ">> publishing node-up.sh to shared storage ($NFS_DIR)"
mkdir -p "$NFS_DIR"
cp "$NODE_UP" "$NFS_DIR/node-up.sh"
chmod +x "$NFS_DIR/node-up.sh"

echo ">> starting Ray HEAD on $HEAD_IP"
bash "$NFS_DIR/node-up.sh" head "$HEAD_IP" "$HEAD_IP"

echo ">> waiting for head GCS to come up..."
sleep 8

for w in "${WORKERS[@]}"; do
  echo ">> starting Ray WORKER on $w"
  ssh -o BatchMode=yes "$SSH_USER@$w" "bash $NFS_DIR/node-up.sh worker $HEAD_IP $w"
done

echo ">> waiting for all 8 GPUs to register with Ray..."
ok=0
for _ in $(seq 1 48); do
  total="$($DOCKER exec "$NAME" ray status 2>/dev/null \
            | grep -oE '/[0-9]+\.0 GPU' | head -1 | grep -oE '[0-9]+' | head -1 || true)"
  echo "   GPUs registered: ${total:-0}/8"
  if [ "${total:-0}" = "8" ]; then ok=1; break; fi
  sleep 5
done
if [ "$ok" != "1" ]; then
  echo "!! cluster did not reach 8 GPUs. Inspect: docker logs $NAME  (and on each worker)." >&2
  exit 1
fi

echo ">> copying serve_vllm.py into the head container"
$DOCKER cp "$HERE/serve_vllm.py" "$NAME:/tmp/serve_vllm.py"

echo ">> launching Ray Serve LLM app inside the head container"
$DOCKER exec -d "$NAME" bash -lc \
  "RAY_ADDRESS=$HEAD_IP:6379 python /tmp/serve_vllm.py > /tmp/serve_vllm.log 2>&1"

cat <<EOF

>> Serve app starting. It will take a few minutes to load 70B across 8 nodes.
   Watch load progress  :  docker exec $NAME tail -f /tmp/serve_vllm.log
   Ready when this works:  curl http://$HEAD_IP:8000/v1/models
   Ray dashboard         :  http://$HEAD_IP:8265
   Serve dashboard       :  http://$HEAD_IP:8265/#/serve
   Tear everything down  :  ../6.vllm/docker-multinode/cluster-down.sh
EOF
