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

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# Model: LLaMA-3.1-8B-Instruct — 8B parameters, ~96 GB training footprint (model + Adam states)
# Single 24GB GPU cannot train this. Requires FSDP across multiple GPUs.
from transformers import AutoModelForCausalLM, AutoTokenizer

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

MODEL_PATH = "/home/user/projects/vllm-deployment/vllm/models/3.1-8b-instruct"
SEQ_LEN    = 512   # token sequence length per sample


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticTextDataset(Dataset):
    """Synthetic token dataset — avoids any download dependency.

    Generates random token IDs in the model's vocabulary range.
    Sufficient to demonstrate FSDP2 sharding and distributed training.
    Replace with a real dataset (e.g. WikiText, OpenWebText) for actual training.
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

def init_model() -> torch.nn.Module:
    """Initialize LLaMA-3.1-8B-Instruct for causal language modelling.

    Stats:
        Parameters:      8,030,000,000
        Model size fp32: ~32.1 GB
        Model size fp16: ~16.1 GB
        Adam states:     ~96.4 GB  (3× model in fp32)
        Total training:  ~128 GB   → cannot fit on a single 24 GB GPU

    Loaded from local path — no internet download required.

    Returns:
        torch.nn.Module: LLaMA-3.1-8B-Instruct model
    """
    logger.info(f"Initializing LLaMA-3.1-8B-Instruct from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,         # load in fp16 — halves RAM: 32GB → 16GB per node
        local_files_only=True,       # never try to download — use local copy
    )
    return model


# ── FSDP2 Sharding ────────────────────────────────────────────────────────────

def shard_model(model: torch.nn.Module):
    """Apply FSDP2 sharding to LLaMA-3.1-8B.

    LLaMA's transformer blocks live at model.model.layers.
    Each block is sharded independently, then the full model wrapper is sharded.
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

    # Shard each transformer decoder block independently
    for decoder_block in model.model.layers:
        fully_shard(
            decoder_block,
            mesh=mesh,
            reshard_after_forward=True,
            offload_policy=offload_policy,
            mp_policy=mp_policy,
        )

    # Shard the full model wrapper
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
    logger.info("Loading distributed checkpoint for resuming training...")
    try:
        with ckpt.as_directory() as checkpoint_dir:
            app_state = AppState(model, optimizer)
            state_dict = {"app": app_state}
            dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)
        logger.info(f"Successfully loaded checkpoint from epoch {app_state.epoch}")
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
    logger.info("Saving checkpoint and reporting metrics...")
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        state_dict = {"app": AppState(model, optimizer, epoch)}
        dcp.save(state_dict=state_dict, checkpoint_id=temp_checkpoint_dir)
        checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
        ray.train.report(metrics, checkpoint=checkpoint)
    logger.info(f"Checkpoint saved successfully. Metrics: {metrics}")


def save_model_for_inference(model: FSDPModule, world_rank: int) -> None:
    logger.info("Preparing model for inference...")
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        save_file = os.path.join(temp_checkpoint_dir, "full-model.pt")
        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
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
    """Main training function — LLaMA-3.1-8B-Instruct with FSDP2 on Ray Train."""

    model = init_model()

    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    # Do NOT call model.to(device) here — FSDP2 moves shards to GPU during fully_shard()
    # Calling .to(device) before sharding would load the full 16GB fp16 model onto one
    # GPU before it gets split, causing immediate OOM on a 24GB card.

    shard_model(model)

    optimizer = Adam(model.parameters(), lr=config.get("learning_rate", 1e-5))

    start_epoch = 0
    loaded_checkpoint = ray.train.get_checkpoint()
    if loaded_checkpoint:
        latest_epoch = load_fsdp_checkpoint(model, optimizer, loaded_checkpoint)
        start_epoch = latest_epoch + 1 if latest_epoch is not None else 0
        logger.info(f"Resuming training from epoch {start_epoch}")

    # Synthetic dataset — 1000 samples of 512 tokens each
    # LLaMA-3.1 vocab size is 128256
    train_data = SyntheticTextDataset(
        vocab_size=128256,
        seq_len=config.get("seq_len", SEQ_LEN),
        num_samples=1000,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.get("batch_size", 1),
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
        epochs = config.get("epochs", 2)

        for epoch in range(start_epoch, epochs):
            if ray.train.get_context().get_world_size() > 1:
                train_loader.sampler.set_epoch(epoch)

            for batch_idx, input_ids in enumerate(train_loader):
                # Causal LM: labels = input_ids (predict next token)
                # The model computes cross-entropy loss internally
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prof.step()

                running_loss += loss.item()
                num_batches += 1

                # progress log every 10 batches, rank 0 only
                if world_rank == 0 and batch_idx % 10 == 0:
                    vram = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx+1}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"VRAM: {vram:.2f} GB"
                    )

            avg_loss = running_loss / num_batches
            metrics = {"loss": avg_loss, "perplexity": torch.exp(torch.tensor(avg_loss)).item()}
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
        "epochs":        2,
        "learning_rate": 1e-5,
        "batch_size":    1,     # OPT-2.7B with seq_len=512: batch=1 per GPU to fit 24GB VRAM
        "seq_len":       512,
    }

    experiment_name = f"llama31_8b_fsdp_{uuid.uuid4().hex[:8]}"

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

    print("Starting LLaMA-3.1-8B-Instruct FSDP2 training job...")
    result = trainer.fit()
    print("Training completed successfully!")

    # ── Inference ─────────────────────────────────────────────────────────────

    PATH_TO_FULL_MODEL = (
        f"/mnt/cluster_storage/{experiment_name}/full_model/full-model.pt"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = init_model()
    state_dict = torch.load(PATH_TO_FULL_MODEL, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    prompt = "The future of distributed AI training is"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(output[0], skip_special_tokens=True))