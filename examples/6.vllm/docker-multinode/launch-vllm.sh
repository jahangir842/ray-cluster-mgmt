#!/usr/bin/env bash
# Launch (or relaunch) the vLLM server inside the head container.
# Idempotent: kills any existing `vllm serve` process in the container first.
# RUN THIS ON THE HEAD NODE.
#
# Factored out of cluster-up.sh so watchdog.sh can call the exact same launch
# command after a crash instead of duplicating it.
set -euo pipefail

HEAD_IP="192.168.3.73"
NAME="${VLLM_CONTAINER:-vllm-ray}"
MODEL="/models/DeepSeek-R1-70B"
PP="8"

DOCKER="docker"; $DOCKER info >/dev/null 2>&1 || DOCKER="sudo docker"

$DOCKER exec "$NAME" pkill -f "vllm serve" 2>/dev/null || true
sleep 2

echo ">> launching vLLM server (PP=$PP) inside $NAME"
$DOCKER exec -d "$NAME" bash -lc \
  "VLLM_USE_V1=0 RAY_ADDRESS=$HEAD_IP:6379 vllm serve $MODEL \
     --served-model-name /models/DeepSeek-R1-70B /home/user/projects/vllm-deployment/vllm/models/3.1-8b-instruct deepseek-r1-70b \
     --tensor-parallel-size 1 --pipeline-parallel-size $PP \
     --distributed-executor-backend ray \
     --gpu-memory-utilization 0.96 --max-model-len 6144 \
     --enforce-eager --host 0.0.0.0 --port 8000 \
     > /tmp/vllm.log 2>&1"
