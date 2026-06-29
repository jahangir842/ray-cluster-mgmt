#!/usr/bin/env bash
# Bring up the multi-node vLLM-on-Ray cluster in Docker and start serving.
# RUN THIS ON THE HEAD NODE (192.168.3.73).
#
# Prereq: the bare-metal Ray cluster must be stopped first (it holds port 6379).
#         Run ./stop-baremetal-ray.sh once before the first Docker bring-up.
set -euo pipefail

HEAD_IP="192.168.3.73"
WORKERS=(192.168.3.71 192.168.3.72 192.168.3.74 192.168.3.75 192.168.3.76 192.168.3.77 192.168.3.78)
SSH_USER="user"
NAME="vllm-ray"
NFS_DIR="/mnt/cluster_storage/vllm-ray"
MODEL="/models/DeepSeek-R1-70B"
PP="8"
HERE="$(cd "$(dirname "$0")" && pwd)"

DOCKER="docker"; $DOCKER info >/dev/null 2>&1 || DOCKER="sudo docker"

echo ">> publishing node-up.sh to shared storage ($NFS_DIR)"
mkdir -p "$NFS_DIR"
cp "$HERE/node-up.sh" "$NFS_DIR/node-up.sh"
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

echo ">> launching vLLM server (PP=$PP) inside the head container"
$DOCKER exec -d "$NAME" bash -lc \
  "VLLM_USE_V1=0 RAY_ADDRESS=$HEAD_IP:6379 vllm serve $MODEL \
     --served-model-name /models/DeepSeek-R1-70B /home/user/projects/vllm-deployment/vllm/models/3.1-8b-instruct deepseek-r1-70b \
     --tensor-parallel-size 1 --pipeline-parallel-size $PP \
     --distributed-executor-backend ray \
     --gpu-memory-utilization 0.96 --max-model-len 6144 \
     --enforce-eager --host 0.0.0.0 --port 8000 \
     > /tmp/vllm.log 2>&1"

cat <<EOF

>> Server starting. It will take a few minutes to load 70B across 8 nodes.
   Watch load progress :  docker exec $NAME tail -f /tmp/vllm.log
   Ready when this works:  curl http://$HEAD_IP:8000/v1/models
   Ray dashboard         :  http://$HEAD_IP:8265
   Tear everything down  :  ./cluster-down.sh
EOF
