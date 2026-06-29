.DEFAULT_GOAL := help

# Load .env if present so variables are available in recipes
-include .env
export

# ─── Help ───────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "Ray Cluster — Docker Compose"
	@echo "════════════════════════════════════════════"
	@echo ""
	@echo "Single-host (dev / CI):"
	@echo "  make build            Build the Ray image"
	@echo "  make up               Start head + workers (WORKER_REPLICAS from .env)"
	@echo "  make down             Stop and remove containers"
	@echo "  make scale N=5        Scale worker count to N"
	@echo "  make logs             Follow head node logs"
	@echo "  make status           Print cluster resource summary"
	@echo "  make shell            Open a shell in the head container"
	@echo "  make job              Run example-job.py on the cluster"
	@echo "  make clean            Remove containers, volumes, and local images"
	@echo ""
	@echo "Multi-node (physical cluster):"
	@echo "  make head-up          Start head container on this machine"
	@echo "  make head-down        Stop head container on this machine"
	@echo "  make worker-up        Start worker container on this machine"
	@echo "  make worker-down      Stop worker container on this machine"
	@echo "  make cluster-up       SSH-deploy to all nodes (reads WORKER_HOSTS)"
	@echo "  make cluster-down     SSH-teardown all nodes"
	@echo "  make cluster-status   ray status from head container"
	@echo ""

# ─── Single-host ────────────────────────────────────────────────────────────
.PHONY: build up down scale logs status shell job clean

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

scale:
	@test -n "$(N)" || (echo "Usage: make scale N=<count>" && exit 1)
	docker compose up -d --scale ray-worker=$(N)

logs:
	docker compose logs -f ray-head

status:
	docker compose exec ray-head ray status

shell:
	docker compose exec ray-head bash

job:
	docker compose exec ray-head python /app/example-job.py

clean:
	docker compose down -v --rmi local

# ─── Multi-node head ────────────────────────────────────────────────────────
.PHONY: head-up head-down

head-up:
	docker compose -f docker-compose.head.yml up -d

head-down:
	docker compose -f docker-compose.head.yml down

# ─── Multi-node worker (run on each worker node) ────────────────────────────
.PHONY: worker-up worker-down

worker-up:
	docker compose -f docker-compose.worker.yml up -d

worker-down:
	docker compose -f docker-compose.worker.yml down

# ─── Cluster-wide SSH deploy ────────────────────────────────────────────────
.PHONY: cluster-up cluster-down cluster-status

cluster-up:
	@bash scripts/deploy-cluster.sh up

cluster-down:
	@bash scripts/deploy-cluster.sh down

cluster-status:
	docker compose -f docker-compose.head.yml exec ray-head ray status
