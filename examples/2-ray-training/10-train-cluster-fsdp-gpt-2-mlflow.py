import os
import io
import tempfile
import uuid
import logging
import math
from pathlib import Path
from typing import Optional

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

from torch.distributed.fsdp import (
    fully_shard,
    FSDPModule,
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

# ── MLflow config ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.3.73:5000")
MLFLOW_EXPERIMENT   = "gpt2-tinystories-3"

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
LOG_FILE = "/tmp/gpt2_training.log"   # captured and uploaded to MLflow at end

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),   # also write to file for MLflow artifact
    ],
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SEQ_LEN        = 1024
TOKENIZER_PATH = "/mnt/cluster_storage/datasets/gpt2_tokenizer"
TOKENIZED_PATH = "/mnt/cluster_storage/datasets/tinystories_tokenized.pt"

# Validation split — 2% of data held out for val loss / accuracy
VAL_SPLIT      = 0.02

# Log training loss to MLflow every N batches (not just at checkpoints)
LOG_EVERY_N_BATCHES = 50


# ── Dataset ───────────────────────────────────────────────────────────────────

class TinyStoriesDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_dataloaders(batch_size: int, seq_len: int):
    """Load, split into train/val, return DataLoaders."""
    logger.info(f"Loading pre-tokenized dataset from {TOKENIZED_PATH} ...")
    raw = torch.load(TOKENIZED_PATH)
    logger.info(f"Loaded {len(raw):,} sequences of length {seq_len}")

    n_val   = int(len(raw) * VAL_SPLIT)
    n_train = len(raw) - n_val

    # deterministic split — same every run
    train_data, val_data = random_split(
        TinyStoriesDataset(raw),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"Train: {n_train:,} sequences | Val: {n_val:,} sequences")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


# ── Model ─────────────────────────────────────────────────────────────────────

def init_model() -> torch.nn.Module:
    from transformers import GPT2Config
    logger.info("Initializing blank GPT-2 from config (no pretrained weights)...")
    config = GPT2Config(
        vocab_size=50257,
        n_positions=SEQ_LEN,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,            # FFN hidden dim = 4 × n_embd
        resid_pdrop=0.1,         # residual dropout
        attn_pdrop=0.1,          # attention dropout
        embd_pdrop=0.1,          # embedding dropout
        loss_type="ForCausalLMLoss",
    )
    model = GPT2LMHeadModel(config)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"GPT-2 initialized — {total_params:,} total params, {trainable_params:,} trainable")
    return model


# ── FSDP2 Sharding ────────────────────────────────────────────────────────────

def shard_model(model: torch.nn.Module):
    logger.info("Applying FSDP2 sharding to model...")
    world_size = ray.train.get_context().get_world_size()
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("data_parallel",),
    )
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    for decoder_block in model.transformer.h:
        fully_shard(decoder_block, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
    logger.info("FSDP2 sharding complete.")


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(model, val_loader, device, max_val_batches: int = 100) -> dict:
    """Compute val loss and token-level accuracy on a subset of val data.

    max_val_batches caps evaluation time — full val set would be slow.
    Token-level accuracy = fraction of tokens where argmax prediction
    matches the ground truth next token.
    """
    model.eval()
    total_loss      = 0.0
    total_correct   = 0
    total_tokens    = 0
    num_batches     = 0

    for batch_idx, input_ids in enumerate(val_loader):
        if batch_idx >= max_val_batches:
            break

        input_ids = input_ids.to(device)
        outputs   = model(input_ids=input_ids, labels=input_ids)
        loss      = outputs.loss

        # Token-level accuracy: shift labels left by 1 (standard causal LM)
        logits  = outputs.logits[:, :-1, :]          # [B, T-1, vocab]
        targets = input_ids[:, 1:]                    # [B, T-1]
        preds   = logits.argmax(dim=-1)               # [B, T-1]

        total_loss    += loss.item()
        total_correct += (preds == targets).sum().item()
        total_tokens  += targets.numel()
        num_batches   += 1

    model.train()

    avg_loss   = total_loss / max(num_batches, 1)
    accuracy   = total_correct / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))   # cap at e^20 to avoid inf

    return {
        "val_loss":       avg_loss,
        "val_perplexity": perplexity,
        "val_accuracy":   accuracy,
    }


# ── Checkpointing ─────────────────────────────────────────────────────────────

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

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        if self.scheduler and state_dict.get("scheduler"):
            self.scheduler.load_state_dict(state_dict["scheduler"])
        self.epoch       = state_dict.get("epoch")
        self.global_step = state_dict.get("global_step", 0)


def load_fsdp_checkpoint(model, optimizer, scheduler, ckpt):
    logger.info("Loading distributed checkpoint for resuming training...")
    try:
        with ckpt.as_directory() as checkpoint_dir:
            app_state = AppState(model, optimizer, scheduler)
            dcp.load(state_dict={"app": app_state}, checkpoint_id=checkpoint_dir)
        logger.info(f"Loaded checkpoint — epoch {app_state.epoch}, step {app_state.global_step}")
        return app_state.epoch, app_state.global_step or 0
    except Exception as e:
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e


def save_checkpoint_and_report(
    model, optimizer, scheduler, metrics,
    epoch=0, global_step=0,
    is_rank0=False,
    mlflow_run_id=None,
    mlflow_client=None,
    checkpoint_dir_on_shared=None,   # if set, also copy checkpoint here and log to MLflow
):
    """Save DCP checkpoint via Ray Train and log metrics to MLflow."""
    logger.info(f"Saving checkpoint at step {global_step}...")

    with tempfile.TemporaryDirectory() as temp_dir:
        dcp.save(
            state_dict={"app": AppState(model, optimizer, scheduler, epoch, global_step)},
            checkpoint_id=temp_dir,
        )
        ray.train.report(
            metrics,
            checkpoint=ray.train.Checkpoint.from_directory(temp_dir),
        )

        # ── Log checkpoint files to MLflow artifacts ──────────────────────────
        if is_rank0 and mlflow_run_id and mlflow_client and checkpoint_dir_on_shared:
            artifact_path = f"checkpoints/step_{global_step:07d}"
            for fname in os.listdir(temp_dir):
                fpath = os.path.join(temp_dir, fname)
                if os.path.isfile(fpath):
                    mlflow_client.log_artifact(mlflow_run_id, fpath, artifact_path=artifact_path)
            logger.info(f"[MLflow] Logged checkpoint files to {artifact_path}")

    # ── Log metrics ───────────────────────────────────────────────────────────
    if is_rank0 and mlflow_run_id and mlflow_client:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow_client.log_metric(mlflow_run_id, k, v, step=global_step)
        logger.info(f"[MLflow] Metrics @ step {global_step}: {metrics}")


def save_model_for_inference(model, world_rank, mlflow_run_id=None, mlflow_client=None):
    logger.info("All-gathering model shards to rank 0 for inference save...")
    with tempfile.TemporaryDirectory() as temp_dir:
        save_file        = os.path.join(temp_dir, "full-model.pt")
        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        checkpoint = None
        if world_rank == 0:
            torch.save(model_state_dict, save_file)
            logger.info(f"Full model saved to {save_file}")
            checkpoint = ray.train.Checkpoint.from_directory(temp_dir)
            if mlflow_run_id and mlflow_client:
                mlflow_client.log_artifact(mlflow_run_id, save_file, artifact_path="full_model")
                logger.info("[MLflow] Logged full-model.pt")
        ray.train.report({}, checkpoint=checkpoint, checkpoint_dir_name="full_model")


# ── Training Function ─────────────────────────────────────────────────────────

def train_func(config):
    """Main training loop — GPT-2 from scratch on TinyStories with FSDP2 + MLflow."""

    ctx        = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    is_rank0   = (world_rank == 0)

    mlflow_run_id = config.get("mlflow_run_id")
    epochs        = config.get("epochs", 2)
    batch_size    = config.get("batch_size", 8)
    lr            = config.get("learning_rate", 1e-5)
    seq_len       = config.get("seq_len", SEQ_LEN)
    val_every     = config.get("val_every_n_batches", 500)   # run val every N batches

    # ── MLflow client (rank 0 only) ───────────────────────────────────────────
    mlflow_client = None
    if is_rank0 and mlflow_run_id:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        logger.info(f"[MLflow] Client ready — run {mlflow_run_id} at {MLFLOW_TRACKING_URI}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = init_model()
    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    shard_model(model)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95)),
        weight_decay=config.get("weight_decay", 0.1),
    )

    # Cosine LR decay: lr decays from lr → lr_min over total training steps
    total_steps = config.get("total_steps_estimate", 7201 * epochs)
    scheduler   = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.get("lr_min", 1e-6),
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch  = 0
    global_step  = 0
    loaded_ckpt  = ray.train.get_checkpoint()
    if loaded_ckpt:
        start_epoch, global_step = load_fsdp_checkpoint(model, optimizer, scheduler, loaded_ckpt)
        start_epoch = (start_epoch or 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}, global_step {global_step}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(batch_size, seq_len)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader   = ray.train.torch.prepare_data_loader(val_loader)

    total_batches = len(train_loader)

    logger.info(
        f"Training | epochs={epochs} | batches/epoch={total_batches} | "
        f"batch_size={batch_size} | seq_len={seq_len} | world_size={world_size} | "
        f"lr={lr} | val_every={val_every} batches"
    )

    # ── Log hyperparams to MLflow (rank 0) ───────────────────────────────────
    if is_rank0 and mlflow_client:
        for k, v in {
            # Model architecture
            "model":            "gpt2-124M-scratch",
            "n_layer":          12,
            "n_head":           12,
            "n_embd":           768,
            "n_inner":          3072,
            "vocab_size":       50257,
            # Training
            "epochs":           epochs,
            "batch_size":       batch_size,
            "seq_len":          seq_len,
            "world_size":       world_size,
            "effective_batch":  batch_size * world_size,   # tokens per step across all GPUs
            "tokens_per_step":  batch_size * world_size * seq_len,
            # Optimizer
            "optimizer":        "Adam",
            "learning_rate":    lr,
            "lr_min":           config.get("lr_min", 1e-6),
            "lr_schedule":      "cosine_annealing",
            "adam_beta1":       config.get("adam_beta1", 0.9),
            "adam_beta2":       config.get("adam_beta2", 0.95),
            "weight_decay":     config.get("weight_decay", 0.1),
            # Regularization
            "resid_pdrop":      0.1,
            "attn_pdrop":       0.1,
            "embd_pdrop":       0.1,
            # Data
            "dataset":          "TinyStories",
            "dataset_path":     TOKENIZED_PATH,
            "tokenizer":        "gpt2",
            "tokenizer_path":   TOKENIZER_PATH,
            "train_sequences":  int(460_813 * (1 - VAL_SPLIT)),
            "val_sequences":    int(460_813 * VAL_SPLIT),
            "val_split":        VAL_SPLIT,
            "val_batches":      100,   # max batches used per val run
            # Infrastructure
            "fsdp_version":     "FSDP2",
            "precision":        "fp32",
            "checkpoint_every": 500,
        }.items():
            mlflow_client.log_param(mlflow_run_id, k, v)

    # ── Profiler ──────────────────────────────────────────────────────────────
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

        running_train_loss = 0.0
        running_batches    = 0

        for epoch in range(start_epoch, epochs):
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            model.train()

            for batch_idx, input_ids in enumerate(train_loader):

                # ── Forward + backward ────────────────────────────────────────
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss    = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                prof.step()

                running_train_loss += loss.item()
                running_batches    += 1
                global_step        += 1

                # ── Per-batch console log ─────────────────────────────────────
                if batch_idx % 10 == 0:
                    vram = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"[Rank {world_rank}] Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx+1}/{total_batches} | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"VRAM: {vram:.2f} GB"
                    )

                # ── Log train loss to MLflow every LOG_EVERY_N_BATCHES ────────
                if is_rank0 and mlflow_client and batch_idx % LOG_EVERY_N_BATCHES == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    mlflow_client.log_metric(
                        mlflow_run_id, "train_loss", loss.item(), step=global_step
                    )
                    mlflow_client.log_metric(
                        mlflow_run_id, "train_loss_smooth",
                        running_train_loss / running_batches, step=global_step
                    )
                    mlflow_client.log_metric(
                        mlflow_run_id, "learning_rate", current_lr, step=global_step
                    )
                    mlflow_client.log_metric(
                        mlflow_run_id, "epoch_progress",
                        epoch + batch_idx / total_batches, step=global_step
                    )

                # ── Validation + checkpoint every val_every batches ───────────
                if batch_idx > 0 and batch_idx % val_every == 0:

                    # Validation (all ranks run it; only rank 0 logs result)
                    val_metrics = run_validation(model, val_loader, device, max_val_batches=100)

                    if is_rank0 and mlflow_client:
                        for k, v in val_metrics.items():
                            mlflow_client.log_metric(mlflow_run_id, k, v, step=global_step)
                        logger.info(f"[MLflow] Val @ step {global_step}: {val_metrics}")

                    # Checkpoint
                    mid_loss = running_train_loss / running_batches
                    save_checkpoint_and_report(
                        model, optimizer, scheduler,
                        metrics={
                            "train_loss":     mid_loss,
                            "train_perplexity": math.exp(min(mid_loss, 20)),
                            **val_metrics,
                            "epoch":          float(epoch),
                        },
                        epoch=epoch, global_step=global_step,
                        is_rank0=is_rank0,
                        mlflow_run_id=mlflow_run_id,
                        mlflow_client=mlflow_client,
                        checkpoint_dir_on_shared=True,
                    )

            # ── End-of-epoch val + checkpoint ─────────────────────────────────
            val_metrics = run_validation(model, val_loader, device, max_val_batches=200)
            avg_loss    = running_train_loss / running_batches

            epoch_metrics = {
                "train_loss":       avg_loss,
                "train_perplexity": math.exp(min(avg_loss, 20)),
                "epoch":            float(epoch + 1),
                **val_metrics,
            }

            save_checkpoint_and_report(
                model, optimizer, scheduler,
                metrics=epoch_metrics,
                epoch=epoch, global_step=global_step,
                is_rank0=is_rank0,
                mlflow_run_id=mlflow_run_id,
                mlflow_client=mlflow_client,
                checkpoint_dir_on_shared=True,
            )
            logger.info(f"Epoch {epoch+1}/{epochs} complete | {epoch_metrics}")

    # ── Memory profiles ───────────────────────────────────────────────────────
    run_name    = ctx.get_experiment_name()
    profile_dir = f"/mnt/cluster_storage/{run_name}"
    os.makedirs(profile_dir, exist_ok=True)

    profile_path = f"{profile_dir}/rank{world_rank}_memory_profile.html"
    prof.export_memory_timeline(profile_path)
    logger.info(f"[Rank {world_rank}] Memory profile saved to {profile_path}")

    if is_rank0 and mlflow_run_id and mlflow_client:
        for rank_id in range(world_size):
            p = f"{profile_dir}/rank{rank_id}_memory_profile.html"
            if os.path.exists(p):
                mlflow_client.log_artifact(mlflow_run_id, p, artifact_path="memory_profiles")
                logger.info(f"[MLflow] Logged memory profile rank {rank_id}")

    # ── Training log file ─────────────────────────────────────────────────────
    if is_rank0 and mlflow_run_id and mlflow_client:
        if os.path.exists(LOG_FILE):
            mlflow_client.log_artifact(mlflow_run_id, LOG_FILE, artifact_path="logs")
            logger.info(f"[MLflow] Logged training log: {LOG_FILE}")

    # ── Full model ────────────────────────────────────────────────────────────
    save_model_for_inference(
        model, world_rank,
        mlflow_run_id=mlflow_run_id,
        mlflow_client=mlflow_client,
    )


# ── Driver / Launch ───────────────────────────────────────────────────────────

if __name__ == "__main__":
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

    experiment_name = f"gpt2_scratch_tinystories_{uuid.uuid4().hex[:8]}"
    run             = mlflow_client_driver.create_run(
        experiment_id=experiment_id,
        run_name=experiment_name,
    )
    mlflow_run_id = run.info.run_id

    # ── Tag the run with metadata ─────────────────────────────────────────────
    for tag_key, tag_val in {
        "mlflow.source.name": __file__,
        "ray.experiment":     experiment_name,
        "cluster.head_ip":    "192.168.3.73",
        "cluster.num_gpus":   "8",
        "cluster.num_nodes":  "8",
    }.items():
        mlflow_client_driver.set_tag(mlflow_run_id, tag_key, tag_val)

    print(f"MLflow run   : {mlflow_run_id}")
    print(f"MLflow UI    : {MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{mlflow_run_id}")
    print()

    train_loop_config = {
        "epochs":               2,
        "learning_rate":        1e-5,
        "lr_min":               1e-6,
        "batch_size":           8,
        "seq_len":              SEQ_LEN,
        "adam_beta1":           0.9,
        "adam_beta2":           0.95,
        "weight_decay":         0.1,
        "val_every_n_batches":  500,    # validation frequency
        "total_steps_estimate": 7201 * 2,
        "mlflow_run_id":        mlflow_run_id,
    }

    scaling_config = ray.train.ScalingConfig(num_workers=8, use_gpu=True)

    RESUME_FROM_CHECKPOINT = None

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
        resume_from_checkpoint=(
            ray.train.Checkpoint.from_directory(RESUME_FROM_CHECKPOINT)
            if RESUME_FROM_CHECKPOINT else None
        ),
    )

    print(f"Starting training | Ray experiment: {experiment_name}")

    try:
        result = trainer.fit()
        print("Training complete!")
        mlflow_client_driver.set_terminated(mlflow_run_id, status="FINISHED")
    except Exception as e:
        mlflow_client_driver.set_terminated(mlflow_run_id, status="FAILED")
        raise

    # ── Inference ─────────────────────────────────────────────────────────────
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