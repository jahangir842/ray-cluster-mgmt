import time
import ray
import ray.train.torch

# ─────────────────────────────────────────────
# BENCHMARK PARAMETERS  (must match single-machine script)
#
# Effective global batch is equal on both sides:
#   Single machine : batch=32, grad_accum=5  → 32×5 = 160 per update
#   Cluster        : batch=32 per worker, 6 workers, grad_accum=1
#                    → 32×6 workers = 192 per update
# ─────────────────────────────────────────────
NUM_WORKERS      = 6           # 6 GPU nodes − 1 head node
PER_WORKER_BATCH = 32          # safe on any GPU ≥ 8GB
GRAD_ACCUM_STEPS = 1           # no accum needed — 6 workers already give 192 global batch
NUM_EPOCHS       = 3
LR               = 0.001
NUM_CLASSES      = 10
DATASET_SIZE     = 3200        # same total samples as single machine
IMAGE_SIZE       = 224
# ─────────────────────────────────────────────

def train_func():
    # All imports MUST be inside train_func for Ray remote workers
    import time
    import torch
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import resnet50
    import ray.train.torch

    rank       = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    if rank == 0:
        print(f"--- Cluster Benchmark | {world_size} workers ---")
        print(f"    Model              : ResNet50")
        print(f"    Image size         : {IMAGE_SIZE}x{IMAGE_SIZE} RGB (synthetic)")
        print(f"    Batch per worker   : {PER_WORKER_BATCH}")
        print(f"    Global batch       : {world_size * PER_WORKER_BATCH}  (= single machine effective batch)")
        print(f"    Grad accum steps   : {GRAD_ACCUM_STEPS}")
        print(f"    Dataset size       : {DATASET_SIZE}\n")

    # Ray automatically moves model to this worker's GPU
    model = resnet50(num_classes=NUM_CLASSES)
    model = ray.train.torch.prepare_model(model)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Each worker gets DATASET_SIZE / world_size samples via DistributedSampler
    # Worker sees: 3200 / 5 = 640 samples → 640 / 32 = 20 steps per epoch
    images     = torch.randn(DATASET_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels     = torch.randint(0, NUM_CLASSES, (DATASET_SIZE,))
    train_data = TensorDataset(images, labels)
    train_loader = DataLoader(
        train_data,
        batch_size=PER_WORKER_BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # Adds DistributedSampler (sharding) + moves batches to this worker's GPU
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    for epoch in range(NUM_EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, (imgs, lbls) in enumerate(train_loader):
            # No .to(device) — prepare_data_loader already handles it
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_time = time.time() - epoch_start

        ray.train.report({
            "epoch":      epoch,
            "loss":       loss.item(),
            "epoch_time": epoch_time,
            "throughput": DATASET_SIZE / epoch_time,
        })

        if rank == 0:
            print(
                f"Epoch {epoch} | loss: {loss.item():.4f} | "
                f"time: {epoch_time:.2f}s | "
                f"throughput: {DATASET_SIZE / epoch_time:.0f} samples/s"
            )

if __name__ == "__main__":
    ray.init(
        address="auto",
        runtime_env={
            "env_vars": {
                "NCCL_SOCKET_IFNAME": "enp0s31f6,enp6s0,eno1",
                "GLOO_SOCKET_IFNAME": "enp0s31f6,enp6s0,eno1",
                "NCCL_IB_DISABLE":    "1",
                "NCCL_P2P_DISABLE":   "1",
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

    print(f"\n--- Cluster Finished ---")
    print(f"Total time     : {total_time:.2f}s")
    print(f"Avg time/epoch : {total_time / NUM_EPOCHS:.2f}s")
    print(f"Final metrics  : {result.metrics}")

    ray.shutdown()