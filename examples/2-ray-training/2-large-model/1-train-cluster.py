import ray
import time
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import ray.train.torch
from transformers import AutoModelForCausalLM, AutoConfig

ray.init(
    address="auto", 
    ignore_reinit_error=True,
    runtime_env={
        "env_vars": {
            "NCCL_SOCKET_IFNAME": "enp0s31f6,eno1",
            "GLOO_SOCKET_IFNAME": "enp0s31f6,eno1",
        }
    }
)

def train_func(config):
    device = ray.train.torch.get_device()
    world_size = ray.train.get_context().get_world_size()
    rank = ray.train.get_context().get_world_rank()
    
    # 1. Initialize the Device Mesh
    device_mesh = init_device_mesh("cuda", (world_size,))
    
    # 2. Load Model Config
    model_id = "/home/user/projects/vllm-deployment/vllm/models/3.1-8b-instruct"
    model_config = AutoConfig.from_pretrained(model_id)
    
    # 3. Create Meta Model and Shard it
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)

    for module in model.modules():
        if "LlamaDecoderLayer" in str(type(module)):
            fully_shard(module, mesh=device_mesh)
    fully_shard(model, mesh=device_mesh)

    # 4. Allocate Real VRAM and Initialize Random Weights for Benchmarking
    model = model.to_empty(device=device)
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-0.01, 0.01)

    print(f"Model successfully sharded and initialized on rank {rank}")

    # --- THE BENCHMARKING LOOP ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create a heavy dummy workload (Batch size 2, Sequence length 512 tokens)
    batch_size = 2
    seq_length = 512
    dummy_input = torch.randint(0, model_config.vocab_size, (batch_size, seq_length), device=device)
    dummy_labels = dummy_input.clone()

    if rank == 0:
        print("\nStarting 8B Parameter FSDP Benchmark...")
        
    start_time = time.time()
    
    # Run 5 heavy training steps
    for step in range(5): 
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(dummy_input, labels=dummy_labels)
        loss = outputs.loss
        
        # Backward pass (This triggers the massive FSDP network syncing)
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Step {step + 1}/5 complete. Loss: {loss.item():.4f}")

    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n--- Benchmark completed in {total_time:.2f} seconds ---")

if __name__ == "__main__":
    scaling_config = ray.train.ScalingConfig(
        num_workers=6, 
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 16}
    )

    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config
    )

    result = trainer.fit()