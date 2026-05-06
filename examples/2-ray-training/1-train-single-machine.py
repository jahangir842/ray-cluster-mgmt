import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

# ─────────────────────────────────────────────
# BENCHMARK PARAMETERS  (must match cluster script)
# ─────────────────────────────────────────────
NUM_EPOCHS        = 3
GLOBAL_BATCH_SIZE = 1024   # same effective batch as 8 workers × 128
LR                = 0.001
NUM_CLASSES       = 10
DATA_DIR          = "/tmp/fashion_mnist_data"
# ─────────────────────────────────────────────

def train_single_machine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Single Machine Benchmark on: {device.type.upper()} ---")
    print(f"    Global batch size : {GLOBAL_BATCH_SIZE}")
    print(f"    Epochs            : {NUM_EPOCHS}")

    start_time = time.time()

    # Model
    model = resnet18(num_classes=NUM_CLASSES)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model = model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Data  — full 60k samples, batch = GLOBAL_BATCH_SIZE
    transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    train_data = FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=GLOBAL_BATCH_SIZE,   # ← equalized to cluster's global batch
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = len(train_loader)
    print(f"    Steps per epoch   : {steps_per_epoch}")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch} | loss: {loss.item():.4f} | "
            f"time: {epoch_time:.2f}s | "
            f"throughput: {len(train_data)/epoch_time:.0f} samples/s"
        )

    total_time = time.time() - start_time
    print("--- Single Machine Training Finished ---")
    print(f"Total time : {total_time:.2f}s")
    print(f"Avg time/epoch : {total_time/NUM_EPOCHS:.2f}s")


if __name__ == "__main__":
    train_single_machine()