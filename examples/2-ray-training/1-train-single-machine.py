import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50

# ─────────────────────────────────────────────
# BENCHMARK PARAMETERS  (must match cluster script)
# ─────────────────────────────────────────────
NUM_EPOCHS        = 3
GLOBAL_BATCH_SIZE = 640        # 5 workers × 128 per worker
LR                = 0.001
NUM_CLASSES       = 10
GRAD_ACCUM_STEPS  = 8          # sync gradients every 8 steps (matches cluster)
DATASET_SIZE      = 6400       # synthetic samples → 10 steps per epoch
IMAGE_SIZE        = 224        # large images → heavy compute per sample
# ─────────────────────────────────────────────


def make_synthetic_dataset(n_samples, image_size, num_classes):
    """
    Synthetic dataset of random 224×224 RGB images.
    Avoids download time so the benchmark measures pure compute.
    """
    images = torch.randn(n_samples, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(images, labels)


def train_single_machine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Single Machine Benchmark on: {device.type.upper()} ---")
    print(f"    Model             : ResNet50")
    print(f"    Image size        : {IMAGE_SIZE}x{IMAGE_SIZE} RGB (synthetic)")
    print(f"    Global batch size : {GLOBAL_BATCH_SIZE}")
    print(f"    Grad accum steps  : {GRAD_ACCUM_STEPS}")
    print(f"    Effective batch   : {GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"    Epochs            : {NUM_EPOCHS}")

    start_time = time.time()

    # ResNet50 — ~25M parameters, much heavier compute than ResNet18
    model = resnet50(num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Synthetic dataset — large 224x224 RGB images
    train_data   = make_synthetic_dataset(DATASET_SIZE, IMAGE_SIZE, NUM_CLASSES)
    train_loader = DataLoader(
        train_data,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = len(train_loader)
    print(f"    Steps per epoch   : {steps_per_epoch} "
          f"(effective weight updates: {steps_per_epoch // GRAD_ACCUM_STEPS})\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            # Divide loss by accum steps so gradients are averaged correctly
            loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS
            loss.backward()

            # Sync and update weights only every GRAD_ACCUM_STEPS
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch} | loss: {(loss.item() * GRAD_ACCUM_STEPS):.4f} | "
            f"time: {epoch_time:.2f}s | "
            f"throughput: {DATASET_SIZE / epoch_time:.0f} samples/s"
        )

    total_time = time.time() - start_time
    print("\n--- Single Machine Training Finished ---")
    print(f"Total time     : {total_time:.2f}s")
    print(f"Avg time/epoch : {total_time / NUM_EPOCHS:.2f}s")


if __name__ == "__main__":
    train_single_machine()