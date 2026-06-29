#!/usr/bin/env bash
# Stop and remove the vLLM-on-Ray Docker containers on every node.
# RUN THIS ON THE HEAD NODE.
set -uo pipefail

HEAD_IP="192.168.3.73"
WORKERS=(192.168.3.71 192.168.3.72 192.168.3.74 192.168.3.75 192.168.3.76 192.168.3.77 192.168.3.78)
SSH_USER="user"
NAME="vllm-ray"

# Try plain docker, fall back to sudo (docker-group membership varies per node).
RM="docker rm -f $NAME 2>/dev/null || sudo docker rm -f $NAME 2>/dev/null || true"

echo ">> removing container on head $HEAD_IP"
sh -c "$RM"

for w in "${WORKERS[@]}"; do
  echo ">> removing container on $w"
  ssh -o BatchMode=yes "$SSH_USER@$w" "$RM"
done

echo ">> all vLLM-Ray containers removed."
