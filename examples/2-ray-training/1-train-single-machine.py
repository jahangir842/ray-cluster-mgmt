import os
import tempfile
import time  # Added time module
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet152, resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

def train():
    # [1] Hardware assignment: Automatically use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # [2] Initialize Model
    # model = resnet18(num_classes=10)
    model = resnet152(num_classes=10)
    # Modify the first conv layer to accept grayscale images (1 channel)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    
    # Move model to the selected device (GPU)
    model = model.to(device)

    # Loss and Optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # [3] Prepare Data
    transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    data_dir = os.path.join(tempfile.gettempdir(), "data")
    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    
    # Standard PyTorch DataLoader
    train_loader = DataLoader(train_data, batch_size=2048, shuffle=True)

    # [4] Standard Training Loop with Timing
    print("Starting training...")
    
    # Start the timer!
    start_time = time.time()
    
    for epoch in range(2):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Move images and labels to the GPU
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Print metrics at the end of each epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/2] - Loss: {avg_loss:.4f}")

    # Stop the timer!
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Format the output into minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"\n--- Training completed in {minutes}m {seconds:.2f}s ---")

    # [5] Save the final model directly
    save_path = "resnet18_fashionmnist.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model state dictionary saved to {save_path}")

if __name__ == "__main__":
    train()