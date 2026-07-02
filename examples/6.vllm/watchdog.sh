#!/usr/bin/env bash
# Auto-recovery watchdog for the BARE-METAL vLLM-on-Ray setup documented in
# 3.vllm-with-ray-backend.md (conda `ray-env`, PP=8 across all 8 nodes, no
# Docker). RUN THIS ON THE HEAD NODE (pc3-4500, 192.168.3.73).
#
# Same failure mode described in that guide's Resilience section: one PP=8
# engine spans every GPU with no spare capacity, so any single worker node's
# Ray actor dying kills the whole `vllm serve` process. There is no live
# failover here -- this only detects the crash and relaunches automatically
# once the flaky node's `ray start` rejoins the head, instead of requiring
# someone to notice and manually redo Steps 5-6 of the guide.
#
# Uses the exact launch command from Step 6 and the exact cleanup from
# Step 5 of 3.vllm-with-ray-backend.md, so it doesn't drift from the
# documented manual procedure.
set -uo pipefail

HEAD_IP="192.168.3.73"
MODEL="$HOME/projects/vllm-deployment/vllm/models/3.1-8b-instruct"
CONDA_ENV="ray-env"
LOG="/tmp/vllm-watchdog.log"

POLL_INTERVAL=15        # seconds between health checks
GPU_WAIT_TIMEOUT=600    # max seconds to wait for a downed node to rejoin before giving up this cycle
RESTART_COOLDOWN=30     # settle time after 8/8 GPUs reappear, before relaunching

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

healthy() {
  curl -sf --max-time 5 "http://$HEAD_IP:8000/v1/models" >/dev/null 2>&1
}

gpus_ready() {
  local total
  total="$(ray status 2>/dev/null | grep -oE '/[0-9]+\.0 GPU' | head -1 | grep -oE '[0-9]+' | head -1 || true)"
  [ "${total:-0}" = "8" ]
}

relaunch() {
  # Step 5 of 3.vllm-with-ray-backend.md: free the port + GPUs first.
  log ">> freeing port 8000 before relaunch"
  pkill -f "vllm serve" 2>/dev/null || true
  sudo fuser -k 8000/tcp 2>/dev/null || true
  sleep 5

  # Step 6: the documented launch command, unchanged.
  log ">> relaunching vllm serve"
  RAY_ADDRESS="$HEAD_IP:6379" \
  VLLM_PLUGINS="" \
  RAY_DEDUP_LOGS=0 \
  nohup vllm serve "$MODEL" \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 8 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --guided-decoding-backend outlines \
    >>"$LOG" 2>&1 &
  disown
}

log ">> watchdog started, polling http://$HEAD_IP:8000/v1/models every ${POLL_INTERVAL}s"

while true; do
  sleep "$POLL_INTERVAL"

  healthy && continue

  log "!! health check failed -- vllm serve is not responding"
  log ">> waiting up to ${GPU_WAIT_TIMEOUT}s for 8/8 GPUs to rejoin the Ray cluster"

  waited=0
  until gpus_ready; do
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge "$GPU_WAIT_TIMEOUT" ]; then
      log "!! gave up waiting for GPUs after ${GPU_WAIT_TIMEOUT}s -- will keep polling health and retry next cycle"
      break
    fi
  done

  gpus_ready || continue   # still short a GPU; loop back to the health check and try again later

  log ">> 8/8 GPUs present, settling ${RESTART_COOLDOWN}s before relaunch"
  sleep "$RESTART_COOLDOWN"

  relaunch

  for _ in $(seq 1 60); do
    if healthy; then
      log ">> vllm serve is healthy again"
      break
    fi
    sleep 10
  done
done
