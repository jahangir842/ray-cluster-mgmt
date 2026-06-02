#!/usr/bin/env bash
# =============================================================================
# start_mlflow_server.sh
#
# Sets up and launches an MLflow tracking server on the Ray head node.
# Artifacts and the SQLite DB are stored in /mnt/cluster_storage/mlflow
# so all Ray workers can reach them via shared storage.
#
# Usage:
#   chmod +x start_mlflow_server.sh
#   ./start_mlflow_server.sh
#
# Stop the server:
#   kill $(cat /tmp/mlflow_server.pid)
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_ROOT="/mnt/cluster_storage/mlflow"
MLFLOW_DB="sqlite:///${MLFLOW_ROOT}/mlflow.db"
MLFLOW_ARTIFACTS="${MLFLOW_ROOT}/artifacts"
MLFLOW_HOST="0.0.0.0"        # bind to all interfaces so workers can reach it
MLFLOW_PORT="5000"
MLFLOW_LOG="/tmp/mlflow_server.log"
MLFLOW_PID="/tmp/mlflow_server.pid"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

log()  { echo -e "${GREEN}[mlflow-setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[mlflow-setup]${NC} $*"; }
die()  { echo -e "${RED}[mlflow-setup] ERROR:${NC} $*" >&2; exit 1; }

# ── 1. Check shared storage is mounted ───────────────────────────────────────
log "Checking shared storage..."
if [[ ! -d /mnt/cluster_storage ]]; then
    die "/mnt/cluster_storage is not mounted. Mount shared storage first."
fi
log "Shared storage OK: /mnt/cluster_storage"

# ── 2. Install / upgrade MLflow ───────────────────────────────────────────────
log "Installing/upgrading MLflow..."
pip install --quiet --upgrade mlflow
MLFLOW_VERSION=$(python3 -c "import mlflow; print(mlflow.__version__)")
log "MLflow version: ${MLFLOW_VERSION}"

# ── 3. Create directories ─────────────────────────────────────────────────────
log "Creating MLflow directories..."
mkdir -p "${MLFLOW_ROOT}"
mkdir -p "${MLFLOW_ARTIFACTS}"
log "  DB path       : ${MLFLOW_DB}"
log "  Artifacts path: ${MLFLOW_ARTIFACTS}"

# ── 4. Kill any existing MLflow server on this port ──────────────────────────
if [[ -f "${MLFLOW_PID}" ]]; then
    OLD_PID=$(cat "${MLFLOW_PID}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        warn "Stopping existing MLflow server (PID ${OLD_PID})..."
        kill "${OLD_PID}" && sleep 2
    fi
    rm -f "${MLFLOW_PID}"
fi

# Also kill anything else sitting on the port
if lsof -ti:"${MLFLOW_PORT}" &>/dev/null; then
    warn "Port ${MLFLOW_PORT} is in use — killing existing process..."
    kill "$(lsof -ti:"${MLFLOW_PORT}")" 2>/dev/null || true
    sleep 2
fi

# ── 5. Detect the head node's IP (workers need this to connect) ───────────────
# Tries Ray's own address first, falls back to hostname -I
HEAD_IP=$(python3 -c "
import ray, os
try:
    ray.init(address='auto', ignore_reinit_error=True)
    ip = ray.get_runtime_context().gcs_address.split(':')[0]
    print(ip)
except Exception:
    import socket
    print(socket.gethostbyname(socket.gethostname()))
" 2>/dev/null || hostname -I | awk '{print $1}')

TRACKING_URI="http://${HEAD_IP}:${MLFLOW_PORT}"

# ── 6. Launch the server ──────────────────────────────────────────────────────
# --serve-artifacts  : lets the UI fetch artifact data through the server
#                      (avoids a second CORS origin for artifact requests)
# MLFLOW_FLASK_SERVER_CORS_ORIGINS=* : sets Access-Control-Allow-Origin so
#                      browsers on any LAN machine can call the REST API
#                      (required for delete/rename actions in the UI)
log "Starting MLflow server..."
MLFLOW_FLASK_SERVER_CORS_ORIGINS="*" \
nohup mlflow server \
    --backend-store-uri     "${MLFLOW_DB}"          \
    --default-artifact-root "${MLFLOW_ARTIFACTS}"   \
    --host                  "${MLFLOW_HOST}"         \
    --port                  "${MLFLOW_PORT}"         \
    --serve-artifacts                                \
    > "${MLFLOW_LOG}" 2>&1 &

SERVER_PID=$!
echo "${SERVER_PID}" > "${MLFLOW_PID}"
log "MLflow server started (PID ${SERVER_PID})"

# ── 7. Wait for the server to become ready (up to 30 s) ──────────────────────
log "Waiting for server to become ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${MLFLOW_PORT}/health" &>/dev/null; then
        log "Server is ready!"
        break
    fi
    if [[ $i -eq 30 ]]; then
        die "Server did not start within 30 s. Check logs: tail ${MLFLOW_LOG}"
    fi
    sleep 1
done

# ── 8. Print summary ──────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  MLflow Tracking Server is RUNNING${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  UI  (open in browser) : ${GREEN}${TRACKING_URI}${NC}"
echo -e "  Tracking URI for code : ${GREEN}${TRACKING_URI}${NC}"
echo -e "  Artifact store        : ${MLFLOW_ARTIFACTS}"
echo -e "  SQLite DB             : ${MLFLOW_ROOT}/mlflow.db"
echo -e "  Server log            : ${MLFLOW_LOG}"
echo -e "  PID file              : ${MLFLOW_PID}"
echo ""
echo -e "  Set in your environment (or training script):"
echo -e "  ${YELLOW}export MLFLOW_TRACKING_URI=${TRACKING_URI}${NC}"
echo ""
echo -e "  Stop the server:"
echo -e "  ${YELLOW}kill \$(cat ${MLFLOW_PID})${NC}"
echo ""