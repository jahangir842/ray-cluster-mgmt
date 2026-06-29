#!/usr/bin/env bash
# Start ONE vLLM-on-Ray container on THIS host. Runs on the node (calls docker).
# Lives on shared NFS so every node executes the identical copy.
#   Usage: node-up.sh <head|worker> <head_ip> <self_ip>
set -euo pipefail

ROLE="${1:?role: head|worker}"
HEAD_IP="${2:?head_ip}"
SELF_IP="${3:?self_ip}"

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.8.0}"
NAME="${VLLM_CONTAINER:-vllm-ray}"
RAY_PORT="6379"
SHM="16g"

# docker-group membership is inconsistent across nodes; passwordless sudo works
# everywhere, so prefer plain docker and fall back to sudo when the socket is denied.
DOCKER="docker"; $DOCKER info >/dev/null 2>&1 || DOCKER="sudo docker"

# Auto-detect the LAN NIC that owns a 192.168.3.x address.
# (enp0s31f6 on the RTX 4500 boxes, eno1 on the RTX 3090 boxes — so we never hardcode.)
IFACE="$(ip -o -4 addr show | grep -F ' 192.168.3.' | awk '{print $2}' | head -1)"
if [ -z "$IFACE" ]; then
  echo "[$SELF_IP] ERROR: no interface owns a 192.168.3.x address" >&2
  exit 1
fi

# Clean any prior container of the same name (idempotent).
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

if [ "$ROLE" = "head" ]; then
  RAY_CMD="start --head --port=${RAY_PORT} --num-gpus=1 --dashboard-host=0.0.0.0 --disable-usage-stats --block"
else
  RAY_CMD="start --address=${HEAD_IP}:${RAY_PORT} --num-gpus=1 --disable-usage-stats --block"
fi

# --network host  : Ray + NCCL use many dynamic ports; host net is the standard for multi-node.
# --shm-size       : Ray plasma store + NCCL live in /dev/shm; the 64MB default crashes.
# --ulimit memlock : pinned memory for NCCL/CUDA.
# NCCL/GLOO ifname : pin THIS node's detected NIC.
# VLLM_HOST_IP     : advertise a unique IP per node (so PP placement sees 8 distinct IPs).
$DOCKER run -d --name "$NAME" --restart unless-stopped \
  --network host --gpus all --shm-size="$SHM" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/cluster_storage/models:/models:ro \
  -e HF_HOME=/tmp/hf \
  -e VLLM_HOST_IP="$SELF_IP" \
  -e NCCL_SOCKET_IFNAME="$IFACE" \
  -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=1 \
  --entrypoint ray "$IMAGE" $RAY_CMD >/dev/null

# The image ships CUDA forward-compat libs. On a node whose host driver is older than
# the image's CUDA (and on consumer GPUs like the RTX 3090s), forward-compat triggers
# CUDA error 804 ("forward compatibility on non supported HW"). Disable it so the
# container uses minor-version compatibility against the host driver. No-op on nodes
# whose driver is already new enough (they never load compat). Driver versions across
# this cluster range 535 -> 595, so this is required for the older nodes.
$DOCKER exec "$NAME" sh -c 'mv /usr/local/cuda/compat /usr/local/cuda/compat.off 2>/dev/null; ldconfig 2>/dev/null' || true

echo "[$SELF_IP] $ROLE container up (image=$IMAGE, iface=$IFACE, gpu=1, via='$DOCKER', compat=disabled)"
