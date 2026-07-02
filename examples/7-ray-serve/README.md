# DeepSeek-R1-70B via Ray Serve LLM

Same multi-node cluster and PP=8 topology as [`../6.vllm/docker-multinode`](../6.vllm/docker-multinode/),
but the model is served through **Ray Serve LLM** (`ray.serve.llm`) instead of
the raw `vllm serve` CLI.

## Why this exists

See the discussion in `6.vllm/docker-multinode/README.md`'s "Resilience"
section. Short version: this cluster runs one pipeline-parallel(8) engine
spanning all 8 GPUs, so there is no spare replica anywhere — losing any one
node still takes the whole (only) replica down, same as the CLI setup.

What changes here: instead of a hand-rolled `watchdog.sh` polling `/v1/models`
and shelling back into the container to relaunch `vllm serve`, **Ray Serve's
controller supervises the replica natively** — it health-checks the
deployment and restarts it when it dies, without a separate script. It's a
supervisor swap, not a redundancy fix.

To get actual node-loss redundancy (traffic keeps flowing while one node is
down), you'd shrink `pipeline_parallel_size` per replica (e.g. a quantized
checkpoint at PP=4) and raise `num_replicas` in `serve_vllm.py`, so that
`num_replicas * pipeline_parallel_size <= 8`. That's a model-quality tradeoff
this repo hasn't made yet — this example is the first step (framework swap)
towards that, not the multi-replica setup itself.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `cluster-up.sh` | head | bring up the same 8-node Ray cluster as `6.vllm/docker-multinode` (reuses its `node-up.sh`), then launch `serve_vllm.py` |
| `serve_vllm.py` | head container | defines the `LLMConfig` (PP=8, `distributed_executor_backend="ray"`) and runs `serve.run(...)` |

Cluster teardown, bare-metal fallback, and per-node NIC/CUDA-compat details
are unchanged — reuse `../6.vllm/docker-multinode/cluster-down.sh` and
`stop-baremetal-ray.sh`.

## Usage

```bash
# on the head (192.168.3.73):
cd ~/projects/ray-cluster-mgmt/examples/7-ray-serve

../6.vllm/docker-multinode/stop-baremetal-ray.sh   # one-time: free 6379 if bare-metal is up
./cluster-up.sh

docker exec vllm-ray tail -f /tmp/serve_vllm.log   # watch 70B load across 8 nodes
curl http://192.168.3.73:8000/v1/models            # ready when this returns the model
```

Ray Serve's own dashboard tab (`http://192.168.3.73:8265/#/serve`) shows
deployment/replica health, which is the piece the CLI + watchdog setup didn't
have natively.

## Dependency note

`ray.serve.llm` ships inside `ray` itself but its runtime path pulls in the
`ray[serve,llm]` extra. The `vllm/vllm-openai:v0.8.0` image used by
`node-up.sh` installs `vllm` (which pins `ray[cgraph]>=2.43.0`) but not
necessarily the `serve` / `llm` extras. If `serve_vllm.py` fails on import
inside the container with a missing-module error, install the extra first:

```bash
docker exec vllm-ray pip install -q "ray[serve,llm]"
```

This hasn't been run against the live cluster yet — treat the first bring-up
as a dry run and watch `serve_vllm.log` closely.
