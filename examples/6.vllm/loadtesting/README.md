# vLLM Load Testing

Two scripts for stress-testing the vLLM OpenAI-compatible server and measuring
latency, throughput, and success rate under concurrent load.

Both are preconfigured for this cluster's deployment:

| Setting | Value |
|---------|-------|
| Server  | `192.168.3.73:8000` (vLLM `--host 0.0.0.0 --port 8000`) |
| Model   | `/home/user/projects/vllm-deployment/vllm/models/3.1-8b-instruct` |
| Endpoint| `/v1/chat/completions` (streaming) |

## Prerequisites

```bash
pip install requests
```

The vLLM server must be running, e.g.:

```bash
RAY_ADDRESS=192.168.3.73:6379 VLLM_PLUGINS="" vllm serve \
  ~/projects/vllm-deployment/vllm/models/3.1-8b-instruct \
  --dtype float16 --tensor-parallel-size 1 --pipeline-parallel-size 7 \
  --distributed-executor-backend ray --gpu-memory-utilization 0.4 \
  --max-model-len 1024 --max-num-seqs 4 --enforce-eager \
  --host 0.0.0.0 --port 8000
```

Sanity-check it's reachable before testing:

```bash
curl -s http://192.168.3.73:8000/v1/models
```

---

## 1. `vllm_load_test.py` — single-machine load test

Spawns N concurrent "users", each sending a sequence of streaming chat requests
with random delays, and prints periodic + final statistics (success rate, avg
response time, P50/P90/P95/P99, throughput, tokens/sec).

```bash
python vllm_load_test.py
```

Tunable constants at the top of the file:

| Constant | Default | Meaning |
|----------|---------|---------|
| `NUM_USERS` | `50` | Concurrent user threads |
| `TEST_DURATION` | `300` | Max test length (seconds) |
| `MESSAGES_PER_USER` | `10` | Messages each user sends before stopping |
| `DELAY_BETWEEN_MESSAGES` | `(3, 8)` | Random think-time range (seconds) |

Output is printed to stdout — periodic snapshots every 10s and a final summary
block.

---

## 2. `distributed_load_test.py` — multi-node load test

Run the **same command on several machines** (each with a unique `--node-id`) to
generate aggregate load far beyond one client's capacity. Each user's full
request/response transcript is written to a per-user log file.

```bash
# on load-gen node 1
python distributed_load_test.py --node-id 1 --users 12 --duration 300

# on load-gen node 2
python distributed_load_test.py --node-id 2 --users 12 --duration 300
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--node-id` | *(required)* | Unique ID per load-gen machine; offsets user IDs |
| `--users` | `12` | Concurrent users on this node |
| `--duration` | `300` | Test length (seconds) |
| `--server` | `192.168.3.73` | Target vLLM host |
| `--port` | `8000` | Target vLLM port |
| `--model` | *(auto)* | Override model ID (else uses the default / first `/v1/models` entry) |
| `--log-dir` | `load_test_logs` | Directory for per-user transcript logs |

This script auto-detects whether the server supports `/v1/chat/completions` or
`/v1/completions`, lists available models, and falls back to the first reported
model if the configured one isn't found.

### Logs

Per-user transcripts land in `--log-dir` as `node<N>_user<M>.txt`, each
containing every query/response pair with timing, token counts, tokens/sec, and
a session summary.

```bash
ls load_test_logs/
cat load_test_logs/node1_user0.txt
```

---

## Interpreting results against the server config

The reference deployment uses `--max-num-seqs 4`, so the engine runs **at most 4
sequences concurrently** — additional requests queue. This is expected and
realistic:

- **High user count (e.g. 50)** → exercises queuing; latency percentiles climb,
  throughput plateaus at the engine's ceiling.
- **Low user count (e.g. 4–8)** → measures near-unqueued per-request latency and
  raw decode throughput.

Run both ends of the range to separate *queue-induced* latency from *model*
latency.

`max_tokens` is `512`, within the server's `--max-model-len 1024` (prompts are
short), so responses are not truncated by the context limit.

## Watching it live in Grafana

Driving load with these scripts populates the **vLLM** Grafana dashboard (see
[`monitoring/`](../../../monitoring/README.md)) — E2E request latency, time to
first token, time per output token, token throughput, scheduler state, and KV
cache utilization. Run a test, then open Grafana → **vLLM** folder and set the
time range to **Last 15 minutes**.
