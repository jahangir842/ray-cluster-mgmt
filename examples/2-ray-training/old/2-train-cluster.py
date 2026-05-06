import os
import ray
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import ray.train.torch

def train_func():
    print(f"--- Worker {ray.train.get_context().get_world_rank()} starting on GPU ---")
    
    # Model Setup
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    # [1] Ray Magic: Because use_gpu=True, this automatically moves the model to the GPU!
    model = ray.train.torch.prepare_model(model)
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Data Setup
    transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    data_dir = "/tmp/fashion_mnist_data" 
    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=2048, shuffle=True)
    
    # [2] Ray Magic: This automatically moves the image batches to the GPU during training!
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    # Training Loop (3 Epochs)
    for epoch in range(3):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        for images, labels in train_loader:
            # We don't even need to write images.to("cuda") because prepare_data_loader did it.
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics = {"loss": loss.item(), "epoch": epoch}
        ray.train.report(metrics)
        
        if ray.train.get_context().get_world_rank() == 0:
            print(f"Epoch {epoch} Complete - Loss: {loss.item():.4f}")

if __name__ == "__main__":
    ray.init(
        address="auto", 
        runtime_env={
            "env_vars": {
                # Tell NCCL: "Look for enp0s31f6 first. If you don't have it, use eno1."
                "NCCL_SOCKET_IFNAME": "enp0s31f6,eno1",
                "GLOO_SOCKET_IFNAME": "enp0s31f6,eno1" 
            }
        }
    )

    # [3] THE BIG UPGRADE: Tell Ray to use 8 workers and demand 1 GPU for each
    scaling_config = ray.train.ScalingConfig(num_workers=7,   use_gpu=True)
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
    )
    
    print("--- Starting GPU-Accelerated ResNet18 Training ---")
    result = trainer.fit()
    print("--- GPU Training Finished Successfully! ---")
    ray.shutdown()