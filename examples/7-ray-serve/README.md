# 3.1-8b-instruct via Ray Serve LLM

Same bare-metal cluster and PP=8 topology as
[`../6.vllm/3.vllm-with-ray-backend.md`](../6.vllm/3.vllm-with-ray-backend.md),
but the model is served through **Ray Serve LLM** (`ray.serve.llm`) instead of
the raw `vllm serve` CLI (Step 6 of that guide).

## Why this exists

See the "Resilience" section in `3.vllm-with-ray-backend.md`. Short version:
this cluster runs one pipeline-parallel(8) engine spanning all 8 GPUs, so
there is no spare replica anywhere — losing any one node still takes the
whole (only) replica down, same as the CLI setup.

What changes here: instead of a hand-rolled `watchdog.sh` polling
`/v1/models` and shelling back in to relaunch `vllm serve`, **Ray Serve's
controller supervises the replica natively** — it health-checks the
deployment and restarts it when it dies, without a separate script. It's a
supervisor swap, not a redundancy fix.

To get actual node-loss redundancy (traffic keeps flowing while one node is
down), you'd shrink `pipeline_parallel_size` per replica and raise
`num_replicas` in `serve_vllm.py`, so that
`num_replicas * pipeline_parallel_size <= 8`. That needs either a smaller
model or a quantized checkpoint to free up GPUs — this example is the first
step (framework swap) towards that, not the multi-replica setup itself.

## Usage

Do Steps 1-5 of `3.vllm-with-ray-backend.md` exactly as documented — env vars
per node, `ray start --head` on pc3, `ray start --address=...` on each
worker, port/GPU freed. Then, instead of Step 6's `vllm serve` command, run
this on the head node:

```bash
conda activate ray-env
cd ~/projects/ray-cluster-mgmt/examples/7-ray-serve

RAY_ADDRESS=192.168.3.73:6379 python serve_vllm.py
```

```bash
curl http://192.168.3.73:8000/v1/models   # ready when this returns the model
```

Ray Serve's own dashboard tab (`http://192.168.3.73:8265/#/serve`) shows
deployment/replica health, which is the piece the plain CLI + watchdog setup
didn't have natively.

To tear down, `Ctrl-C` the script (or `ray stop` on the head, same as the
CLI setup) — cluster bring-up/teardown is otherwise identical to the base
guide.

## Dependency note

`ray.serve.llm` ships inside `ray` itself but its runtime path pulls in the
`ray[serve,llm]` extra. Install it in the `ray-env` conda environment — on
**every** node, not just the head: Ray Serve schedules the deployment replica
wherever it finds resources, and vLLM's own Ray executor places PP workers
across all 8 nodes, so any of them may need to import `ray.llm` internals.

```bash
conda activate ray-env
pip install "ray[serve,llm]"
```

This hasn't been run against the live cluster yet — treat the first run as a
dry run and watch the driver's stdout/`ray status` closely.
