import ray
import torch
import torch.nn as nn
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import ray.train.torch

# 1. The Deep Learning Logic (Standard PyTorch)
def train_loop_per_worker():
    # Create a basic neural network (1 input, 1 output)
    model = nn.Linear(1, 1)
    
    # MAGIC HAPPENS HERE: Ray automatically wraps the model so it can 
    # sync its "brain" over the network with the other machines.
    model = ray.train.torch.prepare_model(model)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Dummy Dataset (The AI needs to learn that the output should be 2x the input)
    inputs = torch.tensor([[1.0], [2.0], [3.0]])
    labels = torch.tensor([[2.0], [4.0], [6.0]])

    # Train for 100 iterations
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
    # Get the ID of the specific worker to prove it's running in parallel
    worker_rank = ray.train.get_context().get_world_rank()
    print(f"[Worker {worker_rank}] Finished training! Final Error (Loss): {loss.item():.4f}")

if __name__ == "__main__":
    # Connect to the cluster
    ray.init(address="auto", ignore_reinit_error=True)
    
    print("--- Configuring Distributed Training ---")
    # 2. The Cluster Configuration
    # We are asking for 4 distributed workers (CPUs since use_gpu=False)
    scaling_config = ScalingConfig(num_workers=4, use_gpu=False)
    
    # 3. The Ray Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
    )
    
    print("--- Starting the Deep Learning Factory ---")
    result = trainer.fit()
    
    print("--- Training Complete! ---")
    ray.shutdown()