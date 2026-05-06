import os
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

def train_single_machine():
    # 1. Device Detection: Automatically use the local GPU if it exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Single Machine Training on: {device.type.upper()} ---")
    
    start_time = time.time()
    
    # 2. Model Setup
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    # Manually push the massive model onto the GPU's memory
    model = model.to(device)
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # 3. Data Setup
    transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    data_dir = "/tmp/fashion_mnist_data" 
    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    
    # Standard PyTorch Dataloader (No Ray Magic)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # 4. Training Loop (3 Epochs)
    for epoch in range(3):
        model.train()
        
        for images, labels in train_loader:
            # Manually push every single batch of images onto the GPU
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Complete - Final Batch Loss: {loss.item():.4f}")

    duration = time.time() - start_time
    print("--- Training Finished Successfully! ---")
    print(f"Total Time: {duration:.2f} seconds")

if __name__ == "__main__":
    train_single_machine()