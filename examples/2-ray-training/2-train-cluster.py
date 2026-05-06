import time
import ray
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import ray.train.torch

# ─────────────────────────────────────────────
# BENCHMARK PARAMETERS  (must match single-machine script)
# ─────────────────────────────────────────────
NUM_EPOCHS         = 3
NUM_WORKERS        = 7          # one per GPU node
PER_WORKER_BATCH   = 147        # 7 × 128 = 896 global batch  ← same as single machine
LR                 = 0.001      # identical LR (same global batch size → no LR scaling needed)
NUM_CLASSES        = 10
DATA_DIR           = "/tmp/fashion_mnist_data"
# ─────────────────────────────────────────────

def train_func():
    rank       = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    if rank == 0:
        print(f"--- Cluster Benchmark | {world_size} workers | "
              f"per-worker batch: {PER_WORKER_BATCH} | "
              f"global batch: {world_size * PER_WORKER_BATCH} ---")

    # Model — Ray moves it to the correct GPU automatically
    model = resnet18(num_classes=NUM_CLASSES)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model = ray.train.torch.prepare_model(model)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Data — DistributedSampler shards 60k samples across all workers
    # Each worker sees 60k / NUM_WORKERS = 7500 samples per epoch
    # Total gradient steps per epoch = 60k / (NUM_WORKERS × PER_WORKER_BATCH) ≈ 59
    transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    train_data = FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=PER_WORKER_BATCH,    # ← per-worker batch
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # prepare_data_loader adds DistributedSampler + moves batches to GPU
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    for epoch in range(NUM_EPOCHS):
        # Required so each epoch gets a different shuffle across workers
        train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.time()

        for images, labels in train_loader:
            # No .to(device) needed — prepare_data_loader handles it
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start

        # Report from every worker; Ray aggregates automatically
        ray.train.report({
            "epoch":      epoch,
            "loss":       loss.item(),
            "epoch_time": epoch_time,
            # Throughput: full dataset / wall-clock epoch time (rank-0 perspective)
            "throughput": len(train_data) / epoch_time,
        })

        if rank == 0:
            print(
                f"Epoch {epoch} | loss: {loss.item():.4f} | "
                f"time: {epoch_time:.2f}s | "
                f"throughput: {len(train_data)/epoch_time:.0f} samples/s"
            )


if __name__ == "__main__":
    ray.init(
    address="auto",
    runtime_env={
        "env_vars": {
            "NCCL_SOCKET_IFNAME":                   "enp0s31f6,eno1",
            "GLOO_SOCKET_IFNAME":                   "enp0s31f6,eno1",
            "RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S": "300",  # ← 5 min timeout
        }
    },
)

    scaling_config = ray.train.ScalingConfig(
        num_workers=NUM_WORKERS,
        use_gpu=True,
    )

    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
    )

    total_start = time.time()
    print("--- Starting Cluster Benchmark ---")
    result = trainer.fit()
    total_time = time.time() - total_start

    print("--- Cluster Training Finished ---")
    print(f"Total time     : {total_time:.2f}s")
    print(f"Avg time/epoch : {total_time/NUM_EPOCHS:.2f}s")
    print(f"Best result    : {result.metrics}")

    ray.shutdown()