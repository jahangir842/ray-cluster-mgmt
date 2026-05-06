import os
import time

# ══════════════════════════════════════════════════════════════════════════════
# BUG 4 FIX — RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S must be set in os.environ
#             BEFORE ray.init(). Setting it inside runtime_env["env_vars"] only
#             reaches remote worker processes — the local controller process
#             (which enforces the timeout) never sees it, so it always fell back
#             to the 60-second default and retried forever.
# ══════════════════════════════════════════════════════════════════════════════
os.environ["RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S"] = "300"

import ray
import ray.train.torch

# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER LAYOUT (confirmed via ray.nodes())
#
#   192.168.3.72  1 GPU      192.168.3.75  1 GPU
#   192.168.3.73  1 GPU  ←── head node (pc3-4500) — also a training worker
#   192.168.3.76  1 GPU      192.168.3.77  1 GPU
#   192.168.3.78  1 GPU
#   Total: 6 nodes × 1 GPU = 6 GPUs
#
# Storage: fully independent per node (no NFS/CIFS shared mount detected).
#   FashionMNIST is downloaded inside train_func to /tmp on each worker node.
#   It is small (~30 MB) and cached after the first run on each node.
#
# ══════════════════════════════════════════════════════════════════════════════
# BUG SUMMARY
#
#   Bug 1 (fatal)   — num_workers=7 but only 6 GPUs → infinite timeout loop
#                     Fix: NUM_WORKERS = 6
#
#   Bug 2 (invalid) — batch_size=2048 cluster vs 128 single machine (16× diff)
#                     Fix: BATCH_SIZE = 128 on both sides
#
#   Bug 3 (crash)   — torch/torchvision imported at module level, not in
#                     train_func. Ray serialises train_func and ships it to
#                     remote workers; workers never re-execute the top of the
#                     file, so those names are undefined → NameError.
#                     Fix: all imports moved inside train_func
#
#   Bug 4 (ignored) — timeout env var in wrong place (see top of file)
#
#   Bug 5 (crash)   — FashionMNIST download attempted simultaneously on 6
#                     independent nodes; if any lacks internet → crash.
#                     Fix: download inside train_func to /tmp (per-node cache);
#                          first run downloads once per node, subsequent runs
#                          are instant.
# ══════════════════════════════════════════════════════════════════════════════

NUM_WORKERS = 6    # BUG 1 FIX: exactly 6 — one per available GPU
BATCH_SIZE  = 128  # BUG 2 FIX: matches single-machine script exactly
NUM_EPOCHS  = 3


def train_func():
    # ── BUG 3 FIX: every import used in this function lives inside it ─────────
    import os
    import time
    import torch
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18
    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import ToTensor, Normalize, Compose
    import ray.train.torch
    # ─────────────────────────────────────────────────────────────────────────

    rank       = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  Cluster Benchmark  |  {world_size} GPU workers")
        print(f"{'='*60}")
        print(f"  Model       : ResNet18  (conv1 patched for 1-channel input)")
        print(f"  Dataset     : FashionMNIST (60 000 training images)")
        print(f"  Batch/worker: {BATCH_SIZE}   ← identical to single-machine script")
        print(f"  Global batch: {BATCH_SIZE * world_size}  (= {BATCH_SIZE} × {world_size} workers)")
        print(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    # Identical construction to the single-machine script.
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    # prepare_model() wraps with DistributedDataParallel and moves to this
    # worker's GPU automatically — no manual .to(device) needed.
    model = ray.train.torch.prepare_model(model)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # ── Dataset ───────────────────────────────────────────────────────────────
    # BUG 5 FIX: download=True is kept, but the download runs inside train_func
    # on each worker node independently.  /tmp is writable on every node and
    # the dataset (~30 MB) is cached there after the first run.
    transform  = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
    data_dir   = "/tmp/fashion_mnist_data"
    train_data = FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,   # keep low inside Ray worker processes to avoid contention
        pin_memory=True,
    )
    # prepare_data_loader() adds a DistributedSampler so each worker sees a
    # unique shard of the data, and moves each batch to this worker's GPU.
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    # ── Training loop ─────────────────────────────────────────────────────────
    total_start = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loader.sampler.set_epoch(epoch)  # different shuffle per epoch
        model.train()
        epoch_start = time.time()

        for images, labels in train_loader:
            # No .to(device) — prepare_data_loader already moved the batch to GPU
            outputs = model(images)
            loss    = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start

        # All workers must call ray.train.report() each epoch
        ray.train.report({
            "epoch":      epoch,
            "loss":       loss.item(),
            "epoch_time": epoch_time,
        })

        if rank == 0:
            print(
                f"  Epoch {epoch} | loss: {loss.item():.4f} | "
                f"time: {epoch_time:.2f}s"
            )

    if rank == 0:
        total = time.time() - total_start
        print(f"\n  Total training time : {total:.2f}s")
        print(f"  Avg time per epoch  : {total / NUM_EPOCHS:.2f}s")


if __name__ == "__main__":
    ray.init(
        address="auto",
        runtime_env={
            "env_vars": {
                # These reach the remote worker processes (correct use of runtime_env).
                # Tells NCCL which network interface to use for GPU-to-GPU communication.
                # Run `ip link show` on a worker node to verify your interface names.
                "NCCL_SOCKET_IFNAME": "enp0s31f6,eno1",
                "GLOO_SOCKET_IFNAME": "enp0s31f6,eno1",
                "NCCL_IB_DISABLE":    "1",  # no InfiniBand on this cluster
                "NCCL_P2P_DISABLE":   "1",  # disable NVLink P2P (cross-node)
            }
        },
    )

    scaling_config = ray.train.ScalingConfig(
        num_workers=NUM_WORKERS,          # BUG 1 FIX: 6 workers = 6 GPUs
        use_gpu=True,
        resources_per_worker={"GPU": 1},  # explicit — skips any CPU-only nodes
    )

    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
    )

    print("--- Starting Cluster Benchmark ---")
    total_start = time.time()
    result = trainer.fit()
    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  Cluster finished")
    print(f"  Total time     : {total_time:.2f}s")
    print(f"  Avg time/epoch : {total_time / NUM_EPOCHS:.2f}s")
    print(f"  Final metrics  : {result.metrics}")
    print(f"{'='*60}")

    ray.shutdown()