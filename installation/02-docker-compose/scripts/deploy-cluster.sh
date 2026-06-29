#!/usr/bin/env bash
# Deploy or tear down the Ray cluster across physical nodes.
#
# Uses your local `ssh3` command for all remote connections.
# Reads configuration from ../.env (relative to this script).
#
# Usage:
#   bash scripts/deploy-cluster.sh up                # head + all workers
#   bash scripts/deploy-cluster.sh up --head-only    # head node only
#   bash scripts/deploy-cluster.sh down              # all nodes
#   bash scripts/deploy-cluster.sh down --head-only  # head node only
#   bash scripts/deploy-cluster.sh install           # install Docker on all nodes
#   bash scripts/deploy-cluster.sh install --head-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$COMPOSE_DIR/.env"

# ── Load .env ────────────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found. Copy .env.example to .env and fill in values."
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

RAY_HEAD_HOST="${RAY_HEAD_HOST:-192.168.3.73}"
SSH_USER="${SSH_USER:-ubuntu}"
WORKER_HOSTS="${WORKER_HOSTS:-}"

# ── Parse args ───────────────────────────────────────────────────────────────
ACTION="${1:-up}"
HEAD_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--head-only" ]] && HEAD_ONLY=true
done

# ── SSH helpers (use your ssh3 command) ──────────────────────────────────────
remote() {
  local host="$1"; shift
  ssh3 "$SSH_USER@$host" "$@"
}

# Copy files to a remote node.
# rsync delegates its SSH transport to ssh3 via --rsh.
# If your ssh3 does not accept the same flags as ssh, switch to the
# ssh3-pipe fallback below by setting USE_SSH3_PIPE=true in .env.
sync_files() {
  local host="$1"
  echo "  → syncing compose files to $host:/opt/ray-cluster/"
  remote "$host" "mkdir -p /opt/ray-cluster"

  if [[ "${USE_SSH3_PIPE:-false}" == "true" ]]; then
    # Fallback: pipe a tar archive over ssh3 when rsync --rsh is not compatible
    tar -czf - \
      -C "$COMPOSE_DIR" \
      Dockerfile .env docker-compose.worker.yml \
      | remote "$host" "tar -xzf - -C /opt/ray-cluster/"
  else
    rsync -az --delete --rsh="ssh3" \
      "$COMPOSE_DIR/Dockerfile" \
      "$COMPOSE_DIR/.env" \
      "$COMPOSE_DIR/docker-compose.worker.yml" \
      "$SSH_USER@$host:/opt/ray-cluster/"
  fi
}

# ── Docker install helper ─────────────────────────────────────────────────────
install_docker_on() {
  local host="$1"
  echo "  → installing Docker on $host"
  remote "$host" bash <<'REMOTE'
    set -e
    if command -v docker &>/dev/null; then
      echo "    Docker already installed: $(docker --version)"
      exit 0
    fi
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    sudo systemctl enable --now docker
    echo "    Docker installed: $(docker --version)"
REMOTE
}

# ── Install action ────────────────────────────────────────────────────────────
if [[ "$ACTION" == "install" ]]; then
  echo "=== Installing Docker on head ($RAY_HEAD_HOST) ==="
  install_docker_on "$RAY_HEAD_HOST"

  if [[ "$HEAD_ONLY" == "false" && -n "$WORKER_HOSTS" ]]; then
    echo ""
    echo "=== Installing Docker on worker nodes ==="
    for HOST in $WORKER_HOSTS; do
      echo "--- $HOST ---"
      install_docker_on "$HOST"
    done
  fi

  echo ""
  echo "=== Docker installation complete ==="
  echo "NOTE: You may need to re-login on each node for the docker group to take effect."
  exit 0
fi

# ── Up action ────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "up" ]]; then
  echo "=== Starting Ray head on $RAY_HEAD_HOST ==="

  # The head node is where this script runs (or the user SSHes into it).
  # We run compose directly here; for a fully remote head, add an ssh3 call.
  docker compose -f "$COMPOSE_DIR/docker-compose.head.yml" build
  docker compose -f "$COMPOSE_DIR/docker-compose.head.yml" up -d

  echo "  → waiting for head to be ready..."
  for i in $(seq 1 12); do
    if docker compose -f "$COMPOSE_DIR/docker-compose.head.yml" \
         exec ray-head ray status &>/dev/null; then
      echo "  ✓ head is ready"
      break
    fi
    echo "  … attempt $i/12"
    sleep 5
  done

  if [[ "$HEAD_ONLY" == "true" ]]; then
    echo ""
    echo "=== Head-only deploy complete ==="
    docker compose -f "$COMPOSE_DIR/docker-compose.head.yml" exec ray-head ray status
    exit 0
  fi

  if [[ -z "$WORKER_HOSTS" ]]; then
    echo "WARNING: WORKER_HOSTS is empty in .env — no workers will be deployed."
    exit 0
  fi

  echo ""
  echo "=== Deploying workers ==="
  for HOST in $WORKER_HOSTS; do
    echo "--- $HOST ---"
    sync_files "$HOST"
    remote "$HOST" "cd /opt/ray-cluster && docker compose -f docker-compose.worker.yml build"
    remote "$HOST" "cd /opt/ray-cluster && docker compose -f docker-compose.worker.yml up -d"
    echo "  ✓ worker started on $HOST"
  done

  echo ""
  echo "=== Cluster is up ==="
  docker compose -f "$COMPOSE_DIR/docker-compose.head.yml" exec ray-head ray status

# ── Down action ───────────────────────────────────────────────────────────────
elif [[ "$ACTION" == "down" ]]; then
  if [[ "$HEAD_ONLY" == "false" && -n "$WORKER_HOSTS" ]]; then
    echo "=== Stopping workers ==="
    for HOST in $WORKER_HOSTS; do
      echo "--- $HOST ---"
      remote "$HOST" \
        "cd /opt/ray-cluster && docker compose -f docker-compose.worker.yml down" || true
      echo "  ✓ $HOST stopped"
    done
    echo ""
  fi

  echo "=== Stopping head ==="
  docker compose -f "$COMPOSE_DIR/docker-compose.head.yml" down
  echo "  ✓ head stopped"

else
  echo "Usage: $0 [up|down|install] [--head-only]"
  exit 1
fi
