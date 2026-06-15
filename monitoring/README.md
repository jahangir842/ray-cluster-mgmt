# Ray Cluster Monitoring — Prometheus + Grafana

Observability stack for the Ray cluster (8 nodes, head at `192.168.3.73`). It
scrapes Ray's per-node metrics agents with **Prometheus** and visualizes them in
**Grafana** using Ray's own auto-generated dashboards. A **node_exporter** is
included for host-level metrics (CPU, RAM, disk, network).

## How it works

Ray exposes Prometheus metrics out of the box:

- Every Ray node runs a metrics agent that exports Prometheus metrics.
- The **head node** writes a service-discovery file to
  `/tmp/ray/prom_metrics_service_discovery.json` listing every node's metrics
  endpoint. Prometheus reads this file (via `file_sd_configs`) so you never
  hard-code node IPs.
- The head also generates Grafana dashboards + provisioning under
  `/tmp/ray/session_latest/metrics/grafana/`. Grafana loads them directly.

Because of this, **run this stack on the Ray head node** (`192.168.3.73`) so
`/tmp/ray` is the real one the head writes to.

```
Ray nodes (metrics agents) ──► Prometheus ──► Grafana ──► (embedded in Ray dashboard)
        ▲ /tmp/ray/prom_metrics_service_discovery.json
```

## Prerequisites

Start the Ray head with a **stable metrics port** so Prometheus targets don't
move on restart:

```bash
ray start --head --metrics-export-port=8080 --dashboard-host=0.0.0.0
```

(Workers also export metrics; the head's discovery file includes them all.)

## Quick start

```bash
cd monitoring
cp .env.example .env          # optional: change Grafana admin password
docker compose up -d
```

| Service      | URL                          | Notes                                  |
|--------------|------------------------------|----------------------------------------|
| Prometheus   | http://192.168.3.73:9090     | Check **Status ▸ Targets** for `ray`   |
| Grafana      | http://192.168.3.73:3000     | login `admin` / `admin` (or your `.env`)|
| node_exporter| http://192.168.3.73:9100     | Raw host metrics                       |

In Grafana, open the **Ray** folder — the cluster / Serve / Train / Data
dashboards are provisioned automatically.

## Embedding Grafana panels in the Ray dashboard (optional)

To see the Grafana charts inside the Ray dashboard at `:8265`, start the Ray
head with these environment variables, then restart Ray:

```bash
export RAY_GRAFANA_HOST=http://192.168.3.73:3000
export RAY_GRAFANA_IFRAME_HOST=http://192.168.3.73:3000
export RAY_PROMETHEUS_HOST=http://192.168.3.73:9090
export RAY_PROMETHEUS_NAME=Prometheus
ray start --head --metrics-export-port=8080 --dashboard-host=0.0.0.0
```

`RAY_GRAFANA_IFRAME_HOST` must be the URL your **browser** uses to reach Grafana.

## vLLM monitoring

vLLM's OpenAI API server exposes Prometheus metrics at `/metrics` on its serving
port (`:8000`) — request throughput, end-to-end latency, time-to-first-token,
time-per-output-token, running vs. waiting requests, and KV-cache usage.

- **Scrape job**: the `vllm` job in `prometheus/prometheus.yml` targets
  `192.168.3.73:8000`. Add more `targets` there if you run several instances.
- **Dashboard**: the official vLLM dashboard ships in
  `grafana/provisioning/dashboards/custom/vllm.json` (patched to use this stack's Prometheus
  datasource). It appears in Grafana's **Custom** folder as "vLLM"; use the
  `model_name` dropdown to pick the served model.

If you run `vllm serve` on a different host/port, update the `vllm` target and
reload Prometheus (`curl -X POST http://localhost:9090/-/reload`).

## Monitoring other cluster nodes' hosts

The included `node_exporter` only covers the head node. For full host metrics,
run a `node_exporter` on each of the other nodes and add them under the
`cluster_node_exporters` job in `prometheus/prometheus.yml`, then reload:

```bash
curl -X POST http://localhost:9090/-/reload
```

## Files

```
monitoring/
├── docker-compose.yml
├── .env.example
├── prometheus/
│   └── prometheus.yml                     # scrape config (Ray SD + node_exporter)
└── grafana/
    └── provisioning/
        ├── datasources/datasource.yml     # Prometheus datasource (uid rayPromDS)
        └── dashboards/
            ├── dashboards.yml             # loads Ray + custom dashboards
            └── custom/
                └── vllm.json             # official vLLM dashboard (datasource pinned)
```

## Troubleshooting

- **`ray` target down / missing in Prometheus** — the head wasn't started with
  `--metrics-export-port`, or `/tmp/ray` inside the container isn't the head's
  real session dir. Confirm `/tmp/ray/prom_metrics_service_discovery.json`
  exists on the host.
- **Grafana "Ray" folder empty** — Ray generates dashboards only after the head
  starts; (re)start Ray, then wait ~30s for Grafana to rescan.
- **Targets show as `down` but file is present** — node IPs in the SD file must
  be reachable from the Prometheus container (they are on the LAN by default).

## Teardown

```bash
docker compose down          # keep data
docker compose down -v       # also wipe Prometheus/Grafana volumes
```
