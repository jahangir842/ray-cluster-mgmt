import os
import tempfile
import uuid
import logging

import torch
import torch.profiler
import torch.distributed.checkpoint as dcp
import ray
import ray.train
import ray.train.torch

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

# Line 14-15 — fix imports
from torchvision.models import vit_h_14, ViT_H_14_Weights, vit_l_32, ViT_L_32_Weights
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Resize, ToTensor, Normalize, Compose

from torch.distributed.fsdp import (
    fully_shard,
    FSDPModule,
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
)
from torch.distributed.device_mesh import init_device_mesh

from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    get_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful

# Enable Ray Train V2 for the latest train APIs
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

_NCCL_ENV = {
    "NCCL_SOCKET_IFNAME": "enp0s31f6,eno1",
    "GLOO_SOCKET_IFNAME": "enp0s31f6,eno1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
os.environ.update(_NCCL_ENV)

# Set up logging
logger = logging.getLogger(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────
def init_model() -> torch.nn.Module:
    """Initialize a Vision Transformer model for FashionMNIST classification.

    Returns:
        torch.nn.Module: Configured ViT model
    """
    logger.info("Initializing Vision Transformer model...")
    model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
    # model = VisionTransformer(
    #     image_size=28,
    #     patch_size=7,
    #     num_layers=10,
    #     num_heads=2,
    #     hidden_dim=128,
    #     mlp_dim=128,
    #     num_classes=10,
    # )

    # Modify patch embedding for grayscale images (1 channel instead of 3)
    model.conv_proj = torch.nn.Conv2d(
        in_channels=1,
        out_channels=1280,
        kernel_size=14,
        stride=14,
    )
    model.heads = torch.nn.Linear(1280, 10)
    return model

# ── FSDP2 Sharding ────────────────────────────────────────────────────────────

def shard_model(model: torch.nn.Module):
    """Apply FSDP2 sharding to the model with optimized configuration.

    Args:
        model: The PyTorch model to shard
    """
    logger.info("Applying FSDP2 sharding to model...")

    world_size = ray.train.get_context().get_world_size()
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("data_parallel",),
    )

    offload_policy = CPUOffloadPolicy()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
    )

    for encoder_block in model.encoder.layers.children():
        fully_shard(
            encoder_block,
            mesh=mesh,
            reshard_after_forward=True,
            offload_policy=offload_policy,
            mp_policy=mp_policy,
        )

    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
        offload_policy=offload_policy,
        mp_policy=mp_policy,
    )

# ── Checkpointing ─────────────────────────────────────────────────────────────

class AppState(Stateful):
    """Stateful wrapper for checkpointing model and optimizer state with DCP."""

    def __init__(self, model, optimizer=None, epoch=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        if "epoch" in state_dict:
            self.epoch = state_dict["epoch"]


def load_fsdp_checkpoint(
    model: FSDPModule,
    optimizer: torch.optim.Optimizer,
    ckpt: ray.train.Checkpoint,
) -> int | None:
    """Load an FSDP checkpoint into the model and optimizer.

    Args:
        model: The FSDP-wrapped model to load state into
        optimizer: The optimizer to load state into
        ckpt: Ray Train checkpoint containing the saved state

    Returns:
        int: The epoch number saved within the checkpoint.
    """
    logger.info("Loading distributed checkpoint for resuming training...")

    try:
        with ckpt.as_directory() as checkpoint_dir:
            app_state = AppState(model, optimizer)
            state_dict = {"app": app_state}
            dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)

        logger.info(f"Successfully loaded distributed checkpoint from epoch {app_state.epoch}")
        return app_state.epoch
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e


def report_metrics_and_save_fsdp_checkpoint(
    model: FSDPModule,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
    epoch: int = 0,
) -> None:
    """Report training metrics and save an FSDP checkpoint.

    Args:
        model: The FSDP-wrapped model to checkpoint
        optimizer: The optimizer to checkpoint
        metrics: Dictionary of metrics to report
        epoch: The current epoch to be saved
    """
    logger.info("Saving checkpoint and reporting metrics...")

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        state_dict = {"app": AppState(model, optimizer, epoch)}
        dcp.save(state_dict=state_dict, checkpoint_id=temp_checkpoint_dir)

        checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
        ray.train.report(metrics, checkpoint=checkpoint)

    logger.info(f"Checkpoint saved successfully. Metrics: {metrics}")


def save_model_for_inference(model: FSDPModule, world_rank: int) -> None:
    """Save the complete unsharded model for inference.

    Args:
        model: The FSDP2-wrapped model to save
        world_rank: The rank of the current worker
    """
    logger.info("Preparing model for inference...")

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        save_file = os.path.join(temp_checkpoint_dir, "full-model.pt")

        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

        logger.info("Successfully retrieved complete model state dict")
        checkpoint = None

        if world_rank == 0:
            torch.save(model_state_dict, save_file)
            logger.info(f"Saved complete model to {save_file}")
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

        ray.train.report({}, checkpoint=checkpoint, checkpoint_dir_name="full_model")


# ── Training Function ─────────────────────────────────────────────────────────

def train_func(config):
    """Main training function that integrates FSDP2 with Ray Train.

    Args:
        config: Training configuration dictionary containing hyperparameters
    """
    model = init_model()

    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    model.to(device)

    shard_model(model)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

    start_epoch = 0
    loaded_checkpoint = ray.train.get_checkpoint()
    if loaded_checkpoint:
        latest_epoch = load_fsdp_checkpoint(model, optimizer, loaded_checkpoint)
        start_epoch = latest_epoch + 1 if latest_epoch is not None else 0
        logger.info(f"Resuming training from epoch {start_epoch}")

    transform = Compose([
        Resize((518, 518)),          # ViT-H/14 requires 518×518
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])

    data_dir = os.path.join(tempfile.gettempdir(), "data")
    train_data = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_data,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
    )
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    world_rank = ray.train.get_context().get_world_rank()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        running_loss = 0.0
        num_batches = 0
        epochs = config.get("epochs", 5)

        for epoch in range(start_epoch, epochs):
            if ray.train.get_context().get_world_size() > 1:
                train_loader.sampler.set_epoch(epoch)

            for images, labels in train_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prof.step()

                running_loss += loss.item()
                num_batches += 1

            avg_loss = running_loss / num_batches
            metrics = {"loss": avg_loss}
            report_metrics_and_save_fsdp_checkpoint(model, optimizer, metrics, epoch)

            if world_rank == 0:
                logger.info(metrics)

    run_name = ray.train.get_context().get_experiment_name()
    prof.export_memory_timeline(
        f"/mnt/cluster_storage/{run_name}/rank{world_rank}_memory_profile.html"
    )

    save_model_for_inference(model, world_rank)


# ── Launch Training ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Connect to the existing cluster and propagate NCCL env vars to all workers
    # via runtime_env — this is the only way to guarantee they are set before
    # any NCCL socket operation on every node.
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env={"env_vars": _NCCL_ENV},
    )

    scaling_config = ray.train.ScalingConfig(
        num_workers=8,
        use_gpu=True,
    )

    train_loop_config = {
        "epochs": 2,
        "learning_rate": 0.001,
        "batch_size": 64,
    }

    experiment_name = f"fsdp_mnist_{uuid.uuid4().hex[:8]}"

    run_config = ray.train.RunConfig(
        storage_path="/mnt/cluster_storage/",
        name=experiment_name,
        failure_config=ray.train.FailureConfig(max_failures=1),
    )

    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    print("Starting FSDP2 training job...")
    result = trainer.fit()
    print("Training completed successfully!")

    # ── Inference ─────────────────────────────────────────────────────────────

    PATH_TO_FULL_MODEL = (
        f"/mnt/cluster_storage/{experiment_name}/full_model/full-model.pt"
    )

    model = init_model()
    state_dict = torch.load(PATH_TO_FULL_MODEL, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    transform = Compose([
        Resize((518, 518)),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])
    test_data = FashionMNIST(root=".", train=False, download=True, transform=transform)

    with torch.no_grad():
        img = transform(test_data.data[0].unsqueeze(0).float() / 255.0).unsqueeze(0)
        out = model(img)
        predicted_label = out.argmax().item()
        test_label = test_data.targets[0].item()
        print(f"{predicted_label=} {test_label=}")