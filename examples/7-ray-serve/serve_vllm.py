"""
Serve DeepSeek-R1-70B with Ray Serve LLM instead of the raw `vllm serve` CLI
used in ../6.vllm/docker-multinode.

Same PP=8 topology (one replica spans all 8 GPUs -- see that folder's README
for why that means zero spare capacity: this is a supervisor swap, not a
redundancy fix). The win over the CLI + watchdog.sh setup is that Ray Serve's
controller supervises the replica itself -- it detects a dead replica and
restarts it without a separate polling script -- and gives the same
OpenAI-compatible HTTP front end via Serve's health-check/readiness path.

To go multi-replica later (real node-loss redundancy, not just faster
recovery): shrink pipeline_parallel_size so each replica fits on fewer GPUs
(e.g. a quantized checkpoint at PP=4) and raise num_replicas so that
num_replicas * pipeline_parallel_size <= 8 GPUs total.
"""
import os

from ray import serve
from ray.serve.llm import LLMConfig, LLMServingArgs, build_openai_app

os.environ.setdefault("VLLM_USE_V1", "0")

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="deepseek-r1-70b",
        model_source="/models/DeepSeek-R1-70B",
    ),
    deployment_config=dict(
        num_replicas=1,
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=8,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.96,
        max_model_len=6144,
        enforce_eager=True,
    ),
)

app = build_openai_app(LLMServingArgs(llm_configs=[llm_config]))

if __name__ == "__main__":
    serve.run(app, blocking=True)
