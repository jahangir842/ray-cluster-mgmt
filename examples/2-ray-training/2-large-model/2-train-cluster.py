import ray
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import ray.train.torch
from transformers import AutoModelForCausalLM, AutoConfig

# 1. Connect to the existing cluster
ray.init(address="auto", ignore_reinit_error=True)

def train_func(config):
    # Initialize Distributed Environment
    device = ray.train.torch.get_device()
    
    # Create Device Mesh (Total 7 GPUs)
    # This mesh is required by FSDP2 for sharding strategy
    device_mesh = init_device_mesh("cuda", (ray.train.torch.get_world_size(),))
    
    # Load Model Configuration only
    model_id = "meta-llama/Meta-Llama-3.1-8b-instruct"
    model_config = AutoConfig.from_pretrained(model_id)
    
    # Initialize model on 'meta' device to prevent OOM
    # This creates the structure without allocating the 15GB+ VRAM yet
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)

    # Apply FSDP2 Sharding to Llama layers
    # Sharding submodules (LlamaDecoderLayer) is crucial for memory efficiency
    for module in model.modules():
        if "LlamaDecoderLayer" in str(type(module)):
            fully_shard(module, mesh=device_mesh)
    
    # Shard the final top-level model
    fully_shard(model, mesh=device_mesh)

    # Note: To start training, you must now load weights into these shards
    # e.g., model.from_pretrained(model_id, low_cpu_mem_usage=True)
    
    print(f"Model successfully sharded on rank {ray.train.torch.get_rank()}")

# 2. Configure Scaling
# We use all 7 GPUs as requested
scaling_config = ray.train.ScalingConfig(
    num_workers=7, 
    use_gpu=True,
    resources_per_worker={"GPU": 1, "CPU": 16}
)

# 3. Launch Trainer
trainer = ray.train.torch.TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config
)

result = trainer.fit()
