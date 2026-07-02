#!/usr/bin/env bash
# Auto-recovery watchdog for the multi-node vLLM-on-Ray cluster.
#
# The cluster runs ONE pipeline-parallel(8) engine spanning all 8 GPUs. If any
# worker node's Ray actor dies, that PP stage is gone with no replacement --
# vllm serve itself dies (this is inherent to vLLM's Ray executor, not a bug
# we can patch around). The Docker container auto-restarts on the node
# (`--restart unless-stopped` in node-up.sh) and rejoins Ray on its own, but
# nothing relaunches `vllm serve` on the head afterwards -- that's what this
# script does.
#
# What this buys you: automatic recovery once the downed node/container comes
# back, instead of a human having to notice and rerun launch-vllm.sh.
# What it does NOT buy you: zero-downtime failover. In-flight requests during
# the outage are lost, and there is no redundancy while the missing GPU is
# down. It also does nothing if the HEAD node itself goes down -- this script
# must be running on the head to relaunch anything.
#
# RUN THIS ON THE HEAD NODE, ideally under systemd (see vllm-watchdog.service).
set -uo pipefail

HEAD_IP="192.168.3.73"
NAME="${VLLM_CONTAINER:-vllm-ray}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/vllm-watchdog.log"

POLL_INTERVAL=15        # seconds between health checks
GPU_WAIT_TIMEOUT=600    # max seconds to wait for a downed node to rejoin before giving up this cycle
RESTART_COOLDOWN=30     # settle time after 8/8 GPUs reappear, before relaunching

DOCKER="docker"; $DOCKER info >/dev/null 2>&1 || DOCKER="sudo docker"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

healthy() {
  curl -sf --max-time 5 "http://$HEAD_IP:8000/v1/models" >/dev/null 2>&1
}

gpus_ready() {
  local total
  total="$($DOCKER exec "$NAME" ray status 2>/dev/null \
            | grep -oE '/[0-9]+\.0 GPU' | head -1 | grep -oE '[0-9]+' | head -1 || true)"
  [ "${total:-0}" = "8" ]
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

  log ">> relaunching vllm serve"
  "$HERE/launch-vllm.sh" >>"$LOG" 2>&1

  for _ in $(seq 1 60); do
    if healthy; then
      log ">> vllm serve is healthy again"
      break
    fi
    sleep 10
  done
done
