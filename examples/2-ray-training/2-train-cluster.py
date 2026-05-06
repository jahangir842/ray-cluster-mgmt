import time
import ray
import ray.train.torch

# ─────────────────────────────────────────────
# BENCHMARK PARAMETERS  (must match single-machine script)
# ─────────────────────────────────────────────
NUM_EPOCHS       = 3
NUM_WORKERS      = 5          # 6 GPU nodes total − 1 head node = 5 worker GPUs
PER_WORKER_BATCH = 128        # 5 × 128 = 640 global batch  ← same as single machine
LR               = 0.001
NUM_CLASSES      = 10
DATA_DIR         = "/tmp/fashion_mnist_data"
# ─────────────────────────────────────────────


def train_func():
    # ── ALL imports must live inside train_func ──────────────────────────────
    # Ray serializes this function and ships it to remote workers.
    # Module-level imports are NOT available on the remote machines.
    # Putting imports here guarantees every worker has them.
    import time
    import torch
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18
    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import ToTensor, Normalize, Compose
    import ray.train.torch
    # ────────────────────────────────────────────────────────────────────────

    rank       = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    if rank == 0:
        print(f"--- Cluster Benchmark | {world_size} workers | "
              f"per-worker batch: {PER_WORKER_BATCH} | "
              f"global batch: {world_size * PER_WORKER_BATCH} ---")

    # Model — Ray moves it to the correct GPU automatically via prepare_model
    model = resnet18(num_classes=NUM_CLASSES)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model = ray.train.torch.prepare_model(model)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Data — DistributedSampler shards 60k samples across all workers
    # Each worker sees 60k / 5 = 12000 samples per epoch
    # Total gradient steps per epoch = 60k / (5 × 128) = 94  ← same as single machine
    transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    train_data = FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=PER_WORKER_BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # Adds DistributedSampler (sharding) + moves batches to the worker's GPU
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    for epoch in range(NUM_EPOCHS):
        train_loader.sampler.set_epoch(epoch)   # different shuffle per epoch
        model.train()
        epoch_start = time.time()

        for images, labels in train_loader:
            # No .to(device) needed — prepare_data_loader already did it
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start

        # ray.train.report() is how workers send metrics back to the driver
        ray.train.report({
            "epoch":      epoch,
            "loss":       loss.item(),
            "epoch_time": epoch_time,
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
                # Cover both machine types in the cluster:
                #   pc3-4500 / pc2-4500 / pc5-4500 → enp0s31f6
                #   pc6-3090 / pc7-3090 / pc8-3090 → enp6s0 or eno1
                "NCCL_SOCKET_IFNAME":  "enp0s31f6,enp6s0,eno1",
                "GLOO_SOCKET_IFNAME":  "enp0s31f6,enp6s0,eno1",
                "NCCL_IB_DISABLE":     "1",    # you're on Ethernet, not InfiniBand
                "NCCL_P2P_DISABLE":    "1",    # no direct GPU peer-to-peer across nodes
                "RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S": "300",
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
    print(f"Final metrics  : {result.metrics}")

    ray.shutdown()