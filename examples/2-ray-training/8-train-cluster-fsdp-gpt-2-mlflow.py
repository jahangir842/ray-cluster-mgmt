import os
import sys
import json
import argparse
import contextlib
import tempfile
import uuid
import logging
import math
from pathlib import Path

import torch
import torch.profiler
import torch.distributed.checkpoint as dcp
import ray
import ray.train
import ray.train.torch
import mlflow
import mlflow.tracking

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from torch.distributed._composable.fsdp import (
    fully_shard,
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

# ── Ray Train V2 ──────────────────────────────────────────────────────────────
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
os.environ["RAY_DEDUP_LOGS"]       = "0"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.3.73:5000")
MLFLOW_EXPERIMENT   = "gpt2-tinystories"

# ── Cluster env — broadcast to all workers via Ray runtime_env ────────────────
_NCCL_ENV = {
    "NCCL_SOCKET_IFNAME":      "enp0s31f6,eno1",
    "GLOO_SOCKET_IFNAME":      "enp0s31f6,eno1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "RAY_DEDUP_LOGS":          "0",
    "MLFLOW_TRACKING_URI":     MLFLOW_TRACKING_URI,
    "HF_HOME":                 "/mnt/cluster_storage/.cache/huggingface",
}
os.environ.update(_NCCL_ENV)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = "/tmp/gpt2_training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

# ── Paths & constants ─────────────────────────────────────────────────────────
SEQ_LEN        = 1024
TOKENIZER_PATH = "/mnt/cluster_storage/datasets/gpt2_tokenizer"
TOKENIZED_PATH = "/mnt/cluster_storage/datasets/tinystories_tokenized.pt"
VAL_SPLIT      = 0.02
LOG_EVERY      = 50
CKPT_EVERY     = 500


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class TinyStoriesDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_dataloaders(batch_size: int, seq_len: int):
    logger.info(f"Loading dataset from {TOKENIZED_PATH} ...")
    raw     = torch.load(TOKENIZED_PATH)
    n_val   = int(len(raw) * VAL_SPLIT)
    n_train = len(raw) - n_val

    train_data, val_data = random_split(
        TinyStoriesDataset(raw),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Dataset — train: {n_train:,} | val: {n_val:,} sequences")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def init_model() -> torch.nn.Module:
    from transformers import GPT2Config
    logger.info("Initializing blank GPT-2 (no pretrained weights)...")
    config = GPT2Config(
        vocab_size  = 50257,
        n_positions = SEQ_LEN,
        n_embd      = 768,
        n_layer     = 12,
        n_head      = 12,
        n_inner     = 3072,
        resid_pdrop = 0.1,
        attn_pdrop  = 0.1,
        embd_pdrop  = 0.1,
        loss_type   = "ForCausalLMLoss",
    )
    model        = GPT2LMHeadModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"GPT-2 ready — {total_params:,} parameters")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# FSDP2 sharding
# ══════════════════════════════════════════════════════════════════════════════

def shard_model(model: torch.nn.Module):
    logger.info("Applying FSDP2 sharding...")
    world_size = ray.train.get_context().get_world_size()
    mesh       = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    mp_policy  = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    for block in model.transformer.h:
        fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
    logger.info("FSDP2 sharding complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_validation(model, val_loader, device, max_batches: int = 100) -> dict:
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
    n_batches     = 0

    for batch_idx, input_ids in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(device)
        outputs   = model(input_ids=input_ids, labels=input_ids)
        logits    = outputs.logits[:, :-1, :]
        targets   = input_ids[:, 1:]
        preds     = logits.argmax(dim=-1)

        total_loss    += outputs.loss.item()
        total_correct += (preds == targets).sum().item()
        total_tokens  += targets.numel()
        n_batches     += 1

    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    return {
        "val_loss":       avg_loss,
        "val_perplexity": math.exp(min(avg_loss, 20)),
        "val_accuracy":   total_correct / max(total_tokens, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

class AppState(Stateful):
    def __init__(self, model, optimizer=None, scheduler=None, epoch=None, global_step=None):
        self.model       = model
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.epoch       = epoch
        self.global_step = global_step

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {
            "model":       model_sd,
            "optim":       optim_sd,
            "scheduler":   self.scheduler.state_dict() if self.scheduler else None,
            "epoch":       self.epoch,
            "global_step": self.global_step,
        }

    def load_state_dict(self, sd):
        set_state_dict(
            self.model, self.optimizer,
            model_state_dict=sd["model"],
            optim_state_dict=sd["optim"],
        )
        if self.scheduler and sd.get("scheduler"):
            self.scheduler.load_state_dict(sd["scheduler"])
        self.epoch       = sd.get("epoch", 0)
        self.global_step = sd.get("global_step", 0)


def load_checkpoint(model, optimizer, scheduler, path: str):
    """Load DCP checkpoint. Returns (epoch, global_step)."""
    logger.info(f"Loading checkpoint from {path} ...")
    ckpt      = ray.train.Checkpoint.from_directory(path)
    app_state = AppState(model, optimizer, scheduler)
    with ckpt.as_directory() as ckpt_dir:
        dcp.load(state_dict={"app": app_state}, checkpoint_id=ckpt_dir)
    logger.info(f"Checkpoint loaded — epoch={app_state.epoch}, global_step={app_state.global_step}")
    return app_state.epoch or 0, app_state.global_step or 0


def save_checkpoint(
    model, optimizer, scheduler, metrics,
    epoch, global_step,
    is_rank0, mlflow_run_id, mlflow_client,
    ray_experiment=None,
):
    logger.info(f"Saving checkpoint at step {global_step}...")

    with tempfile.TemporaryDirectory() as tmp:
        dcp.save(
            state_dict={"app": AppState(model, optimizer, scheduler, epoch, global_step)},
            checkpoint_id=tmp,
        )

        if is_rank0 and mlflow_run_id:
            run_meta = {
                "mlflow_run_id":  mlflow_run_id,
                "ray_experiment": ray_experiment or "",
            }
            with open(os.path.join(tmp, "run_meta.json"), "w") as f:
                json.dump(run_meta, f)

        ray.train.report(
            metrics,
            checkpoint=ray.train.Checkpoint.from_directory(tmp),
        )

        if is_rank0 and mlflow_run_id and mlflow_client:
            artifact_path = f"checkpoints/step_{global_step:07d}"
            for fname in os.listdir(tmp):
                fpath = os.path.join(tmp, fname)
                if os.path.isfile(fpath):
                    mlflow_client.log_artifact(mlflow_run_id, fpath, artifact_path=artifact_path)
            logger.info(f"[MLflow] Checkpoint artifacts → {artifact_path}")

    if is_rank0 and mlflow_run_id and mlflow_client:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow_client.log_metric(mlflow_run_id, k, v, step=global_step)
        logger.info(f"[MLflow] Metrics @ step {global_step}: {metrics}")


def save_full_model(model, world_rank, mlflow_run_id, mlflow_client):
    logger.info("Gathering full model to rank 0...")
    with tempfile.TemporaryDirectory() as tmp:
        save_file = os.path.join(tmp, "full-model.pt")
        model_sd  = get_model_state_dict(
            model=model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        checkpoint = None
        if world_rank == 0:
            torch.save(model_sd, save_file)
            checkpoint = ray.train.Checkpoint.from_directory(tmp)
            if mlflow_run_id and mlflow_client:
                mlflow_client.log_artifact(mlflow_run_id, save_file, artifact_path="full_model")
                logger.info("[MLflow] Logged full-model.pt")
        ray.train.report({}, checkpoint=checkpoint, checkpoint_dir_name="full_model")


# ══════════════════════════════════════════════════════════════════════════════
# MLflow helpers
# ══════════════════════════════════════════════════════════════════════════════

def safe_log_param(client, run_id: str, key: str, value) -> None:
    """Log a param, silently skipping if already logged with the same value."""
    try:
        client.log_param(run_id, key, value)
    except mlflow.exceptions.MlflowException as e:
        if "INVALID_PARAMETER_VALUE" not in str(e):
            raise
        existing = client.get_run(run_id).data.params.get(key)
        if str(existing) != str(value):
            raise ValueError(
                f"[MLflow] Param '{key}' already logged as '{existing}' "
                f"but attempted to log '{value}'."
            ) from e
        logger.debug(f"[MLflow] Skipping duplicate param '{key}' = '{value}' (already logged)")


# ══════════════════════════════════════════════════════════════════════════════
# --inspect mode  (no Ray, no GPU, no training)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(checkpoint_path: str):
    """
    Read epoch + global_step from a DCP checkpoint using a lightweight
    scalar-only proxy — no model weights loaded, no GPU needed.
    Prints the full resume plan so you know exactly where training will resume.
    """

    class _ScalarProxy(Stateful):
        def __init__(self):
            self.epoch       = None
            self.global_step = None
        def state_dict(self):
            return {"epoch": self.epoch, "global_step": self.global_step}
        def load_state_dict(self, sd):
            self.epoch       = sd.get("epoch", 0)
            self.global_step = sd.get("global_step", 0)

    if not os.path.isdir(checkpoint_path):
        print(f"ERROR: checkpoint directory not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nInspecting checkpoint: {checkpoint_path}")
    proxy = _ScalarProxy()
    try:
        dcp.load(state_dict={"app": proxy}, checkpoint_id=checkpoint_path)
    except Exception as e:
        if proxy.epoch is None and proxy.global_step is None:
            raise RuntimeError(f"Could not read scalars from checkpoint: {e}") from e

    epoch       = proxy.epoch       or 0
    global_step = proxy.global_step or 0

    # Read run_meta.json if present
    run_meta_path = os.path.join(checkpoint_path, "run_meta.json")
    mlflow_run_id  = None
    ray_experiment = None
    if os.path.exists(run_meta_path):
        with open(run_meta_path) as f:
            meta = json.load(f)
        mlflow_run_id  = meta.get("mlflow_run_id")
        ray_experiment = meta.get("ray_experiment")

    # Compute resume plan using the same constants as the training script
    batches_per_epoch = 7056        # len(train_loader) at batch_size=8, world_size=8
    batches_done      = global_step % batches_per_epoch
    next_batch        = batches_done
    steps_remaining   = batches_per_epoch - batches_done

    print()
    print("=" * 60)
    print("  CHECKPOINT INSPECTION REPORT")
    print("=" * 60)
    print(f"  Checkpoint path           : {checkpoint_path}")
    print(f"  MLflow run ID             : {mlflow_run_id or '(not found)'}")
    print(f"  Ray experiment            : {ray_experiment or '(not found)'}")
    print("  " + "-" * 56)
    print(f"  Epoch at crash            : {epoch}")
    print(f"  Last completed step       : {global_step}")
    print(f"  Batches per epoch         : {batches_per_epoch}")
    print(f"  Batches already completed : {batches_done}  (= {global_step} % {batches_per_epoch})")
    print(f"  Batches to SKIP on resume : {batches_done}  (indices 0 ... {max(next_batch-1,0)})")
    print(f"  First UNSEEN batch index  : {next_batch}")
    print(f"  Steps remaining in epoch  : {steps_remaining}")
    print("=" * 60)
    print()
    print("Resume training with:")
    print(f"  python {os.path.basename(sys.argv[0])}")
    print("(RAY_EXPERIMENT_NAME in the script must match the experiment above)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Training function (runs on every worker)
# ══════════════════════════════════════════════════════════════════════════════

def train_func(config):
    ctx        = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    is_rank0   = (world_rank == 0)

    epochs        = config.get("epochs", 1)
    batch_size    = config.get("batch_size", 8)
    lr            = config.get("learning_rate", 1e-5)
    seq_len       = config.get("seq_len", SEQ_LEN)
    mlflow_run_id = config.get("mlflow_run_id")
    resume_path   = config.get("resume_checkpoint_path")

    # ── MLflow client (rank 0 only) ───────────────────────────────────────────
    mlflow_client = None
    if is_rank0 and mlflow_run_id:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        logger.info(f"[MLflow] Client ready — run {mlflow_run_id}")

    # ── Model + FSDP ──────────────────────────────────────────────────────────
    model  = init_model()
    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    shard_model(model)

    # ── Optimizer + LR scheduler ──────────────────────────────────────────────
    optimizer = Adam(
        model.parameters(), lr=lr,
        betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95)),
        weight_decay=config.get("weight_decay", 0.1),
    )
    total_steps = config.get("total_steps", 7056 * epochs)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.get("lr_min", 1e-6))

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    is_resumed  = False
    if resume_path:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, resume_path)
        is_resumed = True
        logger.info(f"[MLflow] Resuming into run: {mlflow_run_id}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(batch_size, seq_len)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader   = ray.train.torch.prepare_data_loader(val_loader)
    total_batches = len(train_loader)

    logger.info(
        f"Training | epochs={epochs} | batches/epoch={total_batches} | "
        f"batch_size={batch_size} | seq_len={seq_len} | world_size={world_size} | "
        f"resumed={is_resumed} | start_epoch={start_epoch} | global_step={global_step}"
    )

    # ── Log hyperparams to MLflow (rank 0 only) ───────────────────────────────
    # safe_log_param silently skips params already logged with the same value,
    # preventing INVALID_PARAMETER_VALUE crashes on resume.
    # resumed + resume_from_step change each run so they are logged as metrics.
    if is_rank0 and mlflow_client:
        for k, v in {
            "model":            "gpt2-124M-scratch",
            "n_layer":          12,
            "n_head":           12,
            "n_embd":           768,
            "n_inner":          3072,
            "vocab_size":       50257,
            "resid_pdrop":      0.1,
            "attn_pdrop":       0.1,
            "embd_pdrop":       0.1,
            "epochs":           epochs,
            "batch_size":       batch_size,
            "seq_len":          seq_len,
            "world_size":       world_size,
            "effective_batch":  batch_size * world_size,
            "tokens_per_step":  batch_size * world_size * seq_len,
            "optimizer":        "Adam",
            "learning_rate":    lr,
            "lr_min":           config.get("lr_min", 1e-6),
            "lr_schedule":      "cosine_annealing",
            "adam_beta1":       config.get("adam_beta1", 0.9),
            "adam_beta2":       config.get("adam_beta2", 0.95),
            "weight_decay":     config.get("weight_decay", 0.1),
            "dataset":          "TinyStories",
            "dataset_path":     TOKENIZED_PATH,
            "tokenizer":        "gpt2",
            "tokenizer_path":   TOKENIZER_PATH,
            "train_sequences":  int(460_813 * (1 - VAL_SPLIT)),
            "val_sequences":    int(460_813 * VAL_SPLIT),
            "val_split":        VAL_SPLIT,
            "fsdp_version":     "FSDP2",
            "precision":        "fp32",
            "checkpoint_every": CKPT_EVERY,
            "log_every":        LOG_EVERY,
        }.items():
            safe_log_param(mlflow_client, mlflow_run_id, k, v)

        # resumed + resume_from_step are mutable across resumes → metrics
        mlflow_client.log_metric(mlflow_run_id, "resumed_from_step", float(global_step), step=global_step)
        mlflow_client.log_metric(mlflow_run_id, "is_resumed",        float(is_resumed),  step=global_step)

    # ── Profiler (fresh runs only) ────────────────────────────────────────────
    if not is_resumed:
        prof_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        logger.info("Skipping profiler on resumed run.")
        prof_ctx = contextlib.nullcontext()

    # ── Training loop ─────────────────────────────────────────────────────────
    with prof_ctx as prof:
        running_loss = 0.0
        n_batches    = 0

        for epoch in range(start_epoch, epochs):
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            model.train()

            # Compute how many batches this epoch were already completed
            batches_to_skip = global_step % total_batches if is_resumed else 0

            if batches_to_skip > 0:
                logger.info(
                    f"Fast-forwarding: skipping first {batches_to_skip} batches "
                    f"(indices 0 ... {batches_to_skip - 1}) — already trained on."
                )
                is_resumed = False  # only skip on the first resumed epoch

            for batch_idx, input_ids in enumerate(train_loader):

                # ── EXPLICIT SKIP ─────────────────────────────────────────────
                # Batches 0 ... batches_to_skip-1 were already consumed before
                # the crash. We advance the DataLoader iterator through them
                # but do NOT forward, backward, or step anything.
                if batch_idx < batches_to_skip:
                    if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == batches_to_skip:
                        logger.info(
                            f"  Skipping {batch_idx + 1:,} / {batches_to_skip:,} "
                            f"({100 * (batch_idx + 1) / batches_to_skip:.0f}%)"
                        )
                    continue

                # Forward + backward
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss    = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if prof is not None:
                    prof.step()

                running_loss += loss.item()
                n_batches    += 1
                global_step  += 1

                # Console log every 10 steps
                if global_step % 10 == 0:
                    vram = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"[Rank {world_rank}] Epoch {epoch+1}/{epochs} | "
                        f"Step {global_step} | Batch {batch_idx+1}/{total_batches} | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"VRAM: {vram:.2f} GB"
                    )

                # Log train metrics to MLflow every LOG_EVERY steps
                if is_rank0 and mlflow_client and global_step % LOG_EVERY == 0:
                    smooth = running_loss / n_batches
                    mlflow_client.log_metric(mlflow_run_id, "train_loss",        loss.item(),                 step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "train_loss_smooth", smooth,                      step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "train_perplexity",  math.exp(min(smooth, 20)),   step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "learning_rate",     scheduler.get_last_lr()[0],  step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "epoch_progress",    epoch + batch_idx / total_batches, step=global_step)

                # Validation + checkpoint every CKPT_EVERY steps
                if global_step > 0 and global_step % CKPT_EVERY == 0:
                    val_metrics = run_validation(model, val_loader, device, max_batches=100)

                    if is_rank0 and mlflow_client:
                        for k, v in val_metrics.items():
                            mlflow_client.log_metric(mlflow_run_id, k, v, step=global_step)
                        logger.info(f"[MLflow] Val @ step {global_step}: {val_metrics}")

                    mid_loss = running_loss / n_batches
                    save_checkpoint(
                        model, optimizer, scheduler,
                        metrics={
                            "train_loss":       mid_loss,
                            "train_perplexity": math.exp(min(mid_loss, 20)),
                            "epoch":            float(epoch),
                            **val_metrics,
                        },
                        epoch=epoch, global_step=global_step,
                        is_rank0=is_rank0,
                        mlflow_run_id=mlflow_run_id,
                        mlflow_client=mlflow_client,
                        ray_experiment=ctx.get_experiment_name(),
                    )

            # End-of-epoch: full validation + checkpoint
            val_metrics   = run_validation(model, val_loader, device, max_batches=200)
            avg_loss      = running_loss / n_batches
            epoch_metrics = {
                "train_loss":       avg_loss,
                "train_perplexity": math.exp(min(avg_loss, 20)),
                "epoch":            float(epoch + 1),
                **val_metrics,
            }
            save_checkpoint(
                model, optimizer, scheduler,
                metrics=epoch_metrics,
                epoch=epoch, global_step=global_step,
                is_rank0=is_rank0,
                mlflow_run_id=mlflow_run_id,
                mlflow_client=mlflow_client,
                ray_experiment=ctx.get_experiment_name(),
            )
            logger.info(f"Epoch {epoch+1}/{epochs} complete | {epoch_metrics}")

    # ── Memory profiles (fresh runs only) ─────────────────────────────────────
    run_name    = ctx.get_experiment_name()
    profile_dir = f"/mnt/cluster_storage/{run_name}"
    os.makedirs(profile_dir, exist_ok=True)

    if not is_resumed and prof is not None:
        profile_path = f"{profile_dir}/rank{world_rank}_memory_profile.html"
        try:
            prof.export_memory_timeline(profile_path)
            logger.info(f"[Rank {world_rank}] Memory profile saved.")
        except (ValueError, Exception) as e:
            logger.warning(f"[Rank {world_rank}] Memory profile export skipped: {e}")
            profile_path = None

        if is_rank0 and mlflow_run_id and mlflow_client and profile_path:
            for rank_id in range(world_size):
                p = f"{profile_dir}/rank{rank_id}_memory_profile.html"
                if os.path.exists(p):
                    mlflow_client.log_artifact(mlflow_run_id, p, artifact_path="memory_profiles")
                    logger.info(f"[MLflow] Logged memory profile rank {rank_id}")

    # ── Training log ──────────────────────────────────────────────────────────
    if is_rank0 and mlflow_run_id and mlflow_client and os.path.exists(LOG_FILE):
        mlflow_client.log_artifact(mlflow_run_id, LOG_FILE, artifact_path="logs")
        logger.info("[MLflow] Logged training log.")

    # ── Full model for inference ───────────────────────────────────────────────
    save_full_model(model, world_rank, mlflow_run_id, mlflow_client)


# ══════════════════════════════════════════════════════════════════════════════
# Driver — runs on the head node only
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── --inspect mode: no Ray, no GPU, no training ───────────────────────────
    if "--inspect" in sys.argv:
        parser = argparse.ArgumentParser(
            description="Inspect a DCP checkpoint and print the resume plan."
        )
        parser.add_argument(
            "--inspect",
            metavar="CHECKPOINT_PATH",
            required=True,
            help="Path to the checkpoint directory to inspect.",
        )
        args = parser.parse_args()
        cmd_inspect(args.inspect)
        sys.exit(0)

    # ── Normal training path ──────────────────────────────────────────────────
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env={"env_vars": _NCCL_ENV},
    )

    mlflow_client_driver = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment = mlflow_client_driver.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        experiment_id = mlflow_client_driver.create_experiment(MLFLOW_EXPERIMENT)
        logger.info(f"Created MLflow experiment: {MLFLOW_EXPERIMENT}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using MLflow experiment: {MLFLOW_EXPERIMENT} (id={experiment_id})")

    # ── Resume config ─────────────────────────────────────────────────────────
    RAY_EXPERIMENT_NAME = "gpt2_scratch_tinystories_5c9636bf"
    # RAY_EXPERIMENT_NAME = None   # ← uncomment for a fresh run

    # ── Auto-detect latest checkpoint ─────────────────────────────────────────
    RESUME_FROM_CHECKPOINT = None
    experiment_name        = f"gpt2_scratch_tinystories_{uuid.uuid4().hex[:8]}"

    if RAY_EXPERIMENT_NAME:
        storage_dir = f"/mnt/cluster_storage/{RAY_EXPERIMENT_NAME}"
        if os.path.isdir(storage_dir):
            checkpoints = sorted([
                os.path.join(storage_dir, d)
                for d in os.listdir(storage_dir)
                if d.startswith("checkpoint_") and os.path.isdir(os.path.join(storage_dir, d))
            ])
            if checkpoints:
                RESUME_FROM_CHECKPOINT = checkpoints[-1]
                experiment_name        = RAY_EXPERIMENT_NAME
                logger.info(f"[Resume] Auto-detected latest checkpoint: {RESUME_FROM_CHECKPOINT}")
            else:
                logger.info(f"[Resume] No checkpoints found in {storage_dir} — fresh start")
        else:
            logger.info(f"[Resume] Storage dir not found: {storage_dir} — fresh start")

    # ── MLflow run: reuse on resume, create new on fresh start ────────────────
    mlflow_run_id = None

    if RESUME_FROM_CHECKPOINT:
        run_meta_file = os.path.join(RESUME_FROM_CHECKPOINT, "run_meta.json")
        if os.path.exists(run_meta_file):
            with open(run_meta_file) as f:
                run_meta = json.load(f)
            existing_run_id   = run_meta.get("mlflow_run_id")
            existing_exp_name = run_meta.get("ray_experiment")
            if existing_run_id:
                mlflow_client_driver.update_run(existing_run_id, status="RUNNING")
                mlflow_run_id   = existing_run_id
                experiment_name = existing_exp_name or experiment_name
                logger.info(f"[MLflow] Reactivated existing run: {mlflow_run_id}")
            else:
                logger.info("[MLflow] run_meta.json has no run_id — will create new run")
        else:
            logger.info("[MLflow] No run_meta.json in checkpoint — will create new run")

    if mlflow_run_id is None:
        run           = mlflow_client_driver.create_run(
            experiment_id=experiment_id,
            run_name=experiment_name,
        )
        mlflow_run_id = run.info.run_id
        for k, v in {
            "mlflow.source.name": __file__,
            "ray.experiment":     experiment_name,
            "cluster.head_ip":    "192.168.3.73",
            "cluster.num_gpus":   "8",
            "cluster.num_nodes":  "8",
        }.items():
            mlflow_client_driver.set_tag(mlflow_run_id, k, v)
        logger.info(f"[MLflow] Created new run: {mlflow_run_id}")

    print(f"MLflow run : {mlflow_run_id}")
    print(f"MLflow UI  : {MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{mlflow_run_id}")
    print()

    # ── Training config ───────────────────────────────────────────────────────
    train_loop_config = {
        "epochs":                 1,
        "learning_rate":          1e-5,
        "lr_min":                 1e-6,
        "batch_size":             8,
        "seq_len":                SEQ_LEN,
        "adam_beta1":             0.9,
        "adam_beta2":             0.95,
        "weight_decay":           0.1,
        "total_steps":            7056 * 1,
        "mlflow_run_id":          mlflow_run_id,
        "resume_checkpoint_path": RESUME_FROM_CHECKPOINT,
    }

    scaling_config = ray.train.ScalingConfig(num_workers=8, use_gpu=True)

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

    print(f"Starting training | Ray experiment: {experiment_name}")
    print()

    try:
        result = trainer.fit()
        print("Training complete!")
        mlflow_client_driver.set_terminated(mlflow_run_id, status="FINISHED")
        print(f"MLflow run FINISHED: {mlflow_run_id}")
    except Exception as e:
        mlflow_client_driver.set_terminated(mlflow_run_id, status="FAILED")
        raise

    # ── Inference test ────────────────────────────────────────────────────────
    PATH_TO_FULL_MODEL = f"/mnt/cluster_storage/{experiment_name}/full_model/full-model.pt"
    tokenizer = GPT2Tokenizer.from_pretrained(Path(TOKENIZER_PATH), local_files_only=True)
    model     = init_model()
    model.load_state_dict(torch.load(PATH_TO_FULL_MODEL, map_location="cpu"))
    model.eval()

    prompt = "Once upon a time there was a little girl"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(output[0], skip_special_tokens=True))