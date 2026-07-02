"""
Serve the 3.1-8b-instruct model with Ray Serve LLM instead of the raw
`vllm serve` CLI used in ../6.vllm/3.vllm-with-ray-backend.md.

Run this AFTER Steps 1-5 of that guide (env vars on every node, `ray start
--head` on the head, `ray start --address=...` on each worker, port/GPU
freed) -- it replaces Step 6 only. Same PP=8 topology (one replica spans all
8 GPUs -- see that guide's "Resilience" section for why that means zero spare
capacity: this is a supervisor swap, not a redundancy fix). The win over the
CLI + watchdog.sh setup is that Ray Serve's controller supervises the
replica itself -- it detects a dead replica and restarts it without a
separate polling script -- and gives the same OpenAI-compatible HTTP front
end via Serve's health-check/readiness path.

To go multi-replica later (real node-loss redundancy, not just faster
recovery): shrink pipeline_parallel_size so each replica fits on fewer GPUs
and raise num_replicas so that num_replicas * pipeline_parallel_size <= 8
GPUs total.
"""
import os

from ray import serve
from ray.serve.llm import LLMConfig, LLMServingArgs, build_openai_app

os.environ.setdefault("VLLM_PLUGINS", "")

MODEL_PATH = os.path.expanduser(
    "~/projects/vllm-deployment/vllm/models/3.1-8b-instruct"
)

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="3.1-8b-instruct",
        model_source=MODEL_PATH,
    ),
    deployment_config=dict(
        num_replicas=1,
    ),
    engine_kwargs=dict(
        dtype="float16",
        tensor_parallel_size=1,
        pipeline_parallel_size=8,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enforce_eager=True,
        guided_decoding_backend="outlines",
    ),
)

app = build_openai_app(LLMServingArgs(llm_configs=[llm_config]))

if __name__ == "__main__":
    serve.run(app, blocking=True)
