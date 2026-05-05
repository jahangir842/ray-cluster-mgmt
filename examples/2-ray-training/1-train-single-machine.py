import torch
import torch.nn as nn
import time

def train_single_machine():
    print("--- Starting Single Machine Training ---")
    start_time = time.time()
    
    # 1. The Deep Learning Logic (Standard PyTorch)
    # Create a basic neural network (1 input, 1 output)
    model = nn.Linear(1, 1)
    
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
        
    duration = time.time() - start_time
    print(f"--- Training Complete! ---")
    print(f"Finished in: {duration:.4f} seconds")
    print(f"Final Error (Loss): {loss.item():.4f}")

if __name__ == "__main__":
    train_single_machine()