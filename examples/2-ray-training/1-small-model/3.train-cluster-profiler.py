# Enable Ray Train V2 for the latest train API.
# V2 will be the default in an upcoming release.
import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

# Ray Train imports
import ray.train
import ray.train.torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.models import resnet18
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Compose, ToTensor, Normalize

# # Utility imports
import tempfile
import uuid

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

storage_path = "/mnt/cluster_storage/"

def train_func_distributed():
    """Distributed training function with enhanced profiling for Ray Train."""
    
    # Model, loss, optimizer
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    
    # [1] Prepare model for distributed training.
    # The prepare_model method wraps the model with DistributedDataParallel
    # and moves it to the correct GPU device.
    # ================================================================
    model = ray.train.torch.prepare_model(model)
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Data
    transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    data_dir = os.path.join(tempfile.gettempdir(), "data")
    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    
    # [2] Prepare dataloader for distributed training.
    # The prepare_data_loader method assigns unique rows of data to each worker
    # and handles distributed sampling.
    # ========================================================================
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    # [3] Configure enhanced profiling for distributed training.
    # This includes TensorBoard integration and memory timeline export
    # for comprehensive performance analysis across workers.
    # See more details at https://docs.pytorch.org/docs/stable/profiler.html
    # =============================================================
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{storage_path}/logs/distributed'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        # Training loop
        for epoch in range(10):
            # [4] Set epoch for distributed sampler to ensure proper shuffling
            # across all workers in each epoch.
            # ==============================================================
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            for batch_idx, (images, labels) in enumerate(train_loader):
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prof.step()

                # Log performance metrics every 50 batches
                if batch_idx % 50 == 0 and world_rank == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # [5] Report metrics and checkpoint.
            # Each worker reports its metrics and saves checkpoints to shared storage.
            # ====================================================================
            metrics = {"loss": loss.item(), "epoch": epoch}
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model.pt")
                )
                ray.train.report(
                    metrics,
                    checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
                )
            
            # Log metrics from rank 0 only to avoid duplicate outputs
            if world_rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # [6] Export memory timeline for each worker.
    # This creates separate memory profiles for each worker to analyze
    # memory usage patterns across the distributed training job.
    # ==============================================================
    run_name = ray.train.get_context().get_experiment_name()
    prof.export_memory_timeline(
        f"{storage_path}/{run_name}/rank{world_rank}_memory_profile.html"
    )
    
    if world_rank == 0:
        print(f"Distributed profiling complete! Check '/mnt/cluster_storage/{run_name}/' for worker-specific memory profiles.")
        print("Files generated:")
        print(f"  - rank{world_rank}_memory_profile.html (Memory analysis)")
        print(f"  - rank{world_rank}_chrome_trace.json (Chrome trace)")
        print("  - TensorBoard logs in /mnt/cluster_storage/logs/distributed/")

# Configure scaling and resource requirements for distributed training
scaling_config = ray.train.ScalingConfig(num_workers=6, use_gpu=True)

# Create a unique experiment name for this profiling run
experiment_name = f"profiling_run_{uuid.uuid4().hex[:8]}"

# Configure run settings with persistent storage for profiling outputs.
# The storage_path parameter tells Ray Train where to store experiment artifacts,
# checkpoints, and logs. This is also the same path where PyTorch Profiler outputs
# (TensorBoard traces and memory profiles) are written to, allowing you to access
# all training and profiling results from a single location.
run_config = ray.train.RunConfig(
    storage_path=storage_path,
    name=experiment_name,
)

# Launch distributed training job with profiling
trainer = ray.train.torch.TorchTrainer(
    train_func_distributed,
    scaling_config=scaling_config,
    run_config=run_config,
)

print(f"Starting distributed training with profiling: {experiment_name}")
result = trainer.fit()
print(f"Distributed training with profiling completed successfully! Results are: {result}")
print(f"Check '{storage_path}/{experiment_name}/' for profiling results.")