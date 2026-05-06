import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50

# ─────────────────────────────────────────────
# BENCHMARK PARAMETERS  (must match cluster script)
#
# Memory budget for ResNet50 @ 224x224:
#   batch=32  → ~2GB  (safe on any GPU ≥ 8GB)
#   batch=64  → ~4GB
#   batch=128 → ~8GB
#   batch=640 → ~23GB → OOM on 24GB GPU  ← was the bug
#
# We keep global effective batch = 160 on BOTH sides:
#   Single machine : batch=32, grad_accum=5  → 32×5 = 160 per update
#   Cluster        : batch=32 per worker, 5 workers, grad_accum=1
#                    → 32×5 = 160 per update
# ─────────────────────────────────────────────
BATCH_SIZE       = 32          # per-GPU batch — safe on 24GB GPU
GRAD_ACCUM_STEPS = 6           # simulate 6 workers: 32×6=192 effective batch
NUM_EPOCHS       = 3
LR               = 0.001
NUM_CLASSES      = 10
DATASET_SIZE     = 3200        # synthetic samples → 100 steps per epoch
IMAGE_SIZE       = 224
# ─────────────────────────────────────────────

def train_single_machine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps_per_epoch   = DATASET_SIZE // BATCH_SIZE
    effective_updates = steps_per_epoch // GRAD_ACCUM_STEPS

    print(f"--- Single Machine Benchmark on: {device.type.upper()} ---")
    print(f"    Model              : ResNet50")
    print(f"    Image size         : {IMAGE_SIZE}x{IMAGE_SIZE} RGB (synthetic)")
    print(f"    Batch size (GPU)   : {BATCH_SIZE}")
    print(f"    Grad accum steps   : {GRAD_ACCUM_STEPS}")
    print(f"    Effective batch    : {BATCH_SIZE * GRAD_ACCUM_STEPS}  (= cluster global batch)")
    print(f"    Steps per epoch    : {steps_per_epoch}")
    print(f"    Weight updates/epoch: {effective_updates}\n")

    start_time = time.time()

    model = resnet50(num_classes=NUM_CLASSES).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Synthetic dataset — random 224x224 RGB images, no download needed
    images     = torch.randn(DATASET_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels     = torch.randint(0, NUM_CLASSES, (DATASET_SIZE,))
    train_data = TensorDataset(images, labels)
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            loss = criterion(model(imgs), lbls) / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch} | loss: {loss.item() * GRAD_ACCUM_STEPS:.4f} | "
            f"time: {epoch_time:.2f}s | "
            f"throughput: {DATASET_SIZE / epoch_time:.0f} samples/s"
        )

    total = time.time() - start_time
    print(f"\n--- Single Machine Finished ---")
    print(f"Total time     : {total:.2f}s")
    print(f"Avg time/epoch : {total / NUM_EPOCHS:.2f}s")

if __name__ == "__main__":
    train_single_machine()