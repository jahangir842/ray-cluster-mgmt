import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import ray.train.torch
from transformers import AutoModelForCausalLM, AutoConfig

def train_func(config):
    # 1. Initialize Distributed Environment
    device = torch.device(f"cuda:{ray.train.torch.get_device()}")
    
    # 2. Create Device Mesh (7 GPUs)
    device_mesh = init_device_mesh("cuda", (ray.train.torch.get_world_size(),))
    
    # 3. Load Model on Meta Device (Avoids OOM on rank 0)
    model_id = "meta-llama/Meta-Llama-3-8B"
    model_config = AutoConfig.from_pretrained(model_id)
    
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)

    # 4. Apply FSDP2 Sharding
    # In FSDP2, we shard specific modules (like Transformer blocks)
    for module in model.modules():
        if "LlamaDecoderLayer" in str(type(module)):
            fully_shard(module, mesh=device_mesh)
    
    # Finally, shard the top-level model
    fully_shard(model, mesh=device_mesh)

    # 5. Materialize weights (Sync pre-trained weights to shards)
    # Note: In a real script, you'd use a loader like 'dist_cp' or HF's accelerate
    # to load the actual weights into these shards.
    
    # Standard Training Loop follows...
    print(f"Model sharded on rank {ray.train.torch.get_rank()}")

# 6. Launch on Ray Cluster
scaling_config = ray.train.ScalingConfig(
    num_workers=7, 
    use_gpu=True,
    resources_per_worker={"GPU": 1, "CPU": 4}
)

trainer = ray.train.torch.TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config
)

result = trainer.fit()
