import os
import tempfile
import time  # Added time module
import torch
import ray
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import ray.train.torch

def train_func():
    # Model, Loss, Optimizer
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    
    # [1] Prepare model
    model = ray.train.torch.prepare_model(model)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Data
    transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    data_dir = os.path.join(tempfile.gettempdir(), "data")
    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    
    # [2] Prepare dataloader
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    # Training (Reduced to 2 epochs)
    for epoch in range(2):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # [3] ONLY report metrics (Checkpointing removed to avoid storage error)
        metrics = {"loss": loss.item(), "epoch": epoch}
        ray.train.report(metrics)
            
        if ray.train.get_context().get_world_rank() == 0:
            print(f"Rank 0 reporting -> Epoch: {epoch+1}/2, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # Connect to your existing Ray cluster
    ray.init(
        address="auto",
        runtime_env={
            "env_vars": {
                "NCCL_SOCKET_IFNAME": "enp0s31f6,eno1",
                "GLOO_SOCKET_IFNAME": "enp0s31f6,eno1",
            }
        },
    )

    # [4] Configure scaling (Set to 2 to use your 2x RTX 3090s)
    scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True)

    # [5] Launch distributed training job
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
    )
    
    print("Starting cluster training demo...")
    
    # Start the timer!
    start_time = time.time()
    
    # Run the workload
    result = trainer.fit()

    # Stop the timer!
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Format the output into minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    
    print(f"\n--- Cluster Training completed in {minutes}m {seconds:.2f}s ---")
    print("Demo complete! (Model was not saved to avoid shared storage requirements).")