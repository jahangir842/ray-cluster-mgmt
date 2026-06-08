import os
import json
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

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

from torch.distributed._composable.fsdp import (
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

# ── Ray Train V2 ──────────────────────────────────────────────────────────────
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
os.environ["RAY_DEDUP_LOGS"]       = "0"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.3.73:5000")
MLFLOW_EXPERIMENT   = "llama31-8b-fsdp"

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
LOG_FILE = "/tmp/llama_training.log"
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
MODEL_PATH = "/home/user/projects/vllm-deployment/vllm/models/3.1-8b-instruct"
SEQ_LEN    = 256
VAL_SPLIT  = 0.05    # 5% held out for validation
LOG_EVERY  = 2      # log train_loss to MLflow every N steps
CKPT_EVERY = 50      # checkpoint + validation every N steps

# LLaMA 3.1 vocab size
LLAMA_VOCAB_SIZE = 128256


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

WIKITEXT_PATH = "/mnt/cluster_storage/datasets/wikitext2"


class WikiTextDataset(Dataset):
    """WikiText-2 dataset loaded from shared storage.

    Concatenates all text into one long token stream, then chunks into
    fixed-length sequences of seq_len tokens. This is the standard
    "pack and chunk" approach for language model training — no padding,
    no wasted tokens.

    Expected path: /mnt/cluster_storage/datasets/wikitext2
    Expected format: HuggingFace dataset with a "text" column.
    """
    def __init__(self, tokenizer, seq_len: int, split: str = "train"):
        logger.info(f"Loading WikiText-2 ({split}) from {WIKITEXT_PATH} ...")
        dataset = load_from_disk(WIKITEXT_PATH)[split]

        # Concatenate all non-empty lines into one long string then tokenize
        text   = " ".join([x for x in dataset["text"] if x.strip()])
        tokens = tokenizer.encode(text)

        self.data = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            self.data.append(torch.tensor(tokens[i:i + seq_len]))

        # self.data = self.data[:100] 

        logger.info(
            f"WikiText-2 {split}: {len(tokens):,} tokens → "
            f"{len(self.data):,} sequences of length {seq_len}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_dataloaders(tokenizer, batch_size: int, seq_len: int):
    """Build train and validation DataLoaders from WikiText-2.

    Uses the dataset's own train/validation/test splits instead of
    a random split — this gives comparable results to published benchmarks.
    """
    train_dataset = WikiTextDataset(tokenizer, seq_len, split="train")
    val_dataset   = WikiTextDataset(tokenizer, seq_len, split="validation")

    logger.info(f"Train: {len(train_dataset):,} sequences | Val: {len(val_dataset):,} sequences")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def init_model() -> torch.nn.Module:
    """Initialize LLaMA-3.1-8B-Instruct for causal LM.

    Memory breakdown:
        Parameters:      8,030,000,000
        Model fp16:      ~16 GB
        Adam states:     ~48 GB  (2× model in fp16)
        Activations:     ~4-8 GB per GPU (seq_len=512, batch=1)
        Total training:  ~80-96 GB → requires FSDP across all 8 GPUs
    """
    logger.info(f"Loading LLaMA-3.1-8B-Instruct from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded — {total_params:,} total | {trainable_params:,} trainable params")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# FSDP2 sharding
# ══════════════════════════════════════════════════════════════════════════════

def shard_model(model: torch.nn.Module, cpu_offload: bool = True):
    """Apply FSDP2 sharding to LLaMA-3.1-8B.

    LLaMA's transformer blocks live at model.model.layers (32 layers).
    Each block is sharded independently, then the outer wrapper is sharded.

    cpu_offload=True: keeps optimizer states on CPU to save GPU memory.
    Trades ~30% throughput for 3× lower GPU memory usage.
    Set to False if you have enough VRAM.
    """
    logger.info(f"Applying FSDP2 sharding (cpu_offload={cpu_offload})...")
    world_size = ray.train.get_context().get_world_size()
    mesh       = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    offload_policy = CPUOffloadPolicy() if cpu_offload else None
    mp_policy      = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    for block in model.model.layers:
        fully_shard(
            block,
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
    logger.info("FSDP2 sharding complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_validation(model, val_loader, device, max_batches: int = 20) -> dict:
    """Compute val loss and token-level accuracy."""
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

        logits  = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        preds   = logits.argmax(dim=-1)

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
    """DCP-compatible checkpoint wrapper.

    mlflow_run_id and ray_experiment are intentionally NOT in state_dict()
    to keep DCP key-schema stable across versions. They live in run_meta.json.
    """
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
    """Load DCP checkpoint directly from path. Returns (epoch, global_step)."""
    logger.info(f"Loading checkpoint from {path} ...")
    app_state = AppState(model, optimizer, scheduler)
    dcp.load(state_dict={"app": app_state}, checkpoint_id=path)
    logger.info(f"Checkpoint loaded — epoch={app_state.epoch}, global_step={app_state.global_step}")
    return app_state.epoch or 0, app_state.global_step or 0


def save_checkpoint(
    model, optimizer, scheduler, metrics,
    epoch, global_step,
    is_rank0, mlflow_run_id, mlflow_client,
    ray_experiment=None,
):
    """Save DCP checkpoint, write run_meta.json, report to Ray, log to MLflow."""
    logger.info(f"Saving checkpoint at step {global_step}...")

    with tempfile.TemporaryDirectory() as tmp:
        dcp.save(
            state_dict={"app": AppState(model, optimizer, scheduler, epoch, global_step)},
            checkpoint_id=tmp,
        )

        # run_meta.json — driver reads this on next resume to reuse same MLflow run
        if is_rank0 and mlflow_run_id:
            with open(os.path.join(tmp, "run_meta.json"), "w") as f:
                json.dump({
                    "mlflow_run_id":  mlflow_run_id,
                    "ray_experiment": ray_experiment or "",
                }, f)

        ray.train.report(
            metrics,
            checkpoint=ray.train.Checkpoint.from_directory(tmp),
        )

        # Log checkpoint shard files to MLflow artifacts
        if is_rank0 and mlflow_run_id and mlflow_client:
            artifact_path = f"checkpoints/step_{global_step:07d}"
            for fname in os.listdir(tmp):
                fpath = os.path.join(tmp, fname)
                if os.path.isfile(fpath):
                    mlflow_client.log_artifact(mlflow_run_id, fpath, artifact_path=artifact_path)
            logger.info(f"[MLflow] Checkpoint → {artifact_path}")

    # Log metrics
    if is_rank0 and mlflow_run_id and mlflow_client:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow_client.log_metric(mlflow_run_id, k, v, step=global_step)
        logger.info(f"[MLflow] Metrics @ step {global_step}: {metrics}")


def save_full_model(model, world_rank, mlflow_run_id, mlflow_client):
    """Gather full model to rank 0, save for inference, log to MLflow."""
    logger.info("Gathering full model to rank 0 for inference export...")
    with tempfile.TemporaryDirectory() as tmp:
        save_file = os.path.join(tmp, "full-model.pt")
        model_sd  = get_model_state_dict(
            model=model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        checkpoint = None
        if world_rank == 0:
            torch.save(model_sd, save_file)
            logger.info(f"Full model saved to {save_file}")
            checkpoint = ray.train.Checkpoint.from_directory(tmp)
            if mlflow_run_id and mlflow_client:
                mlflow_client.log_artifact(mlflow_run_id, save_file, artifact_path="full_model")
                logger.info("[MLflow] Logged full-model.pt")
        ray.train.report({}, checkpoint=checkpoint, checkpoint_dir_name="full_model")


# ══════════════════════════════════════════════════════════════════════════════
# Training function (runs on every worker)
# ══════════════════════════════════════════════════════════════════════════════

def train_func(config):
    ctx        = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    is_rank0   = (world_rank == 0)

    epochs        = config.get("epochs", 2)
    batch_size    = config.get("batch_size", 1)
    lr            = config.get("learning_rate", 1e-5)
    seq_len       = config.get("seq_len", SEQ_LEN)
    cpu_offload   = config.get("cpu_offload", True)
    mlflow_run_id = config.get("mlflow_run_id")
    resume_path   = config.get("resume_checkpoint_path")

    # ── MLflow client (rank 0 only) ───────────────────────────────────────────
    mlflow_client = None
    if is_rank0 and mlflow_run_id:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        logger.info(f"[MLflow] Client ready — run {mlflow_run_id}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Load tokenizer on all workers — needed to build WikiText dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model + FSDP ──────────────────────────────────────────────────────────
    model  = init_model()
    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    # Do NOT call model.to(device) before sharding —
    # FSDP2 moves each rank's shard to GPU inside fully_shard()
    shard_model(model, cpu_offload=cpu_offload)

    # ── Optimizer + LR scheduler ──────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95)),
        weight_decay=config.get("weight_decay", 0.1),
        eps=config.get("adam_eps", 1e-8),
    )
    total_steps = config.get("total_steps", 100 * epochs)
    scheduler   = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.get("lr_min", 1e-7),
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    is_resumed  = False
    if resume_path:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, resume_path)
        is_resumed = True
        logger.info(f"[MLflow] Resuming into run: {mlflow_run_id}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(tokenizer, batch_size, seq_len)
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader   = ray.train.torch.prepare_data_loader(val_loader)
    total_batches = len(train_loader)

    logger.info(
        f"Training | epochs={epochs} | batches/epoch={total_batches} | "
        f"batch_size={batch_size} | seq_len={seq_len} | world_size={world_size} | "
        f"resumed={is_resumed} | global_step={global_step}"
    )

    # ── Log hyperparams to MLflow (fresh runs only) ───────────────────────────
    if is_rank0 and mlflow_client and not is_resumed:
        for k, v in {
            # Model architecture
            "model":              "LLaMA-3.1-8B-Instruct",
            "model_path":         MODEL_PATH,
            "num_parameters":     "8,030,000,000",
            "num_layers":         32,
            "num_heads":          32,
            "num_kv_heads":       8,
            "hidden_size":        4096,
            "intermediate_size":  14336,
            "vocab_size":         LLAMA_VOCAB_SIZE,
            "context_length":     8192,
            "rope_theta":         500000.0,
            # Training
            "epochs":             epochs,
            "batch_size":         batch_size,
            "seq_len":            seq_len,
            "world_size":         world_size,
            "effective_batch":    batch_size * world_size,
            "tokens_per_step":    batch_size * world_size * seq_len,
            # Optimizer
            "optimizer":          "AdamW",
            "learning_rate":      lr,
            "lr_min":             config.get("lr_min", 1e-7),
            "lr_schedule":        "cosine_annealing",
            "adam_beta1":         config.get("adam_beta1", 0.9),
            "adam_beta2":         config.get("adam_beta2", 0.95),
            "weight_decay":       config.get("weight_decay", 0.1),
            "adam_eps":           config.get("adam_eps", 1e-8),
            # Infrastructure
            "fsdp_version":       "FSDP2",
            "param_dtype":        "float16",
            "reduce_dtype":       "float16",
            "cpu_offload":        cpu_offload,
            "checkpoint_every":   CKPT_EVERY,
            "log_every":          LOG_EVERY,
            "val_split":          VAL_SPLIT,
            "dataset":            "WikiText-2",
            "dataset_path":       WIKITEXT_PATH,
            "dataset_split":      "train/validation (HF splits)",
            "tokenizer":          "LLaMA-3.1-8B-Instruct tokenizer",
        }.items():
            mlflow_client.log_param(mlflow_run_id, k, v)

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

            # On resume: fast-forward already-completed batches
            batches_to_skip = global_step % total_batches if is_resumed else 0
            if batches_to_skip > 0:
                logger.info(f"Skipping {batches_to_skip} already-completed batches...")
                skipped = 0
                for _ in train_loader:
                    skipped += 1
                    if skipped >= batches_to_skip:
                        break
                logger.info(f"Fast-forwarded {batches_to_skip} batches. Resuming from batch {batches_to_skip+1}.")
                is_resumed = False

            for batch_idx, input_ids in enumerate(train_loader):
                batch_idx = batch_idx + batches_to_skip

                # Forward + backward — causal LM loss
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss    = outputs.loss

                optimizer.zero_grad()
                loss.backward()

                grad_norm = torch.tensor(1.0)

                optimizer.step()
                scheduler.step()

                if prof is not None:
                    prof.step()

                running_loss += loss.item()
                n_batches    += 1
                global_step  += 1

                # Console log every 5 steps
                if global_step % 5 == 0:
                    vram = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"[Rank {world_rank}] Epoch {epoch+1}/{epochs} | "
                        f"Step {global_step} | Batch {batch_idx+1}/{total_batches} | "
                        f"Loss: {loss.item():.4f} | "
                        f"GradNorm: {grad_norm:.3f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"VRAM: {vram:.2f} GB"
                    )

                # Log train metrics to MLflow every LOG_EVERY steps
                if is_rank0 and mlflow_client and global_step % LOG_EVERY == 0:
                    smooth = running_loss / n_batches
                    cur_lr = scheduler.get_last_lr()[0]
                    mlflow_client.log_metric(mlflow_run_id, "train_loss",        loss.item(),              step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "train_loss_smooth",  smooth,                   step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "train_perplexity",   math.exp(min(smooth, 20)), step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "learning_rate",      cur_lr,                   step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "grad_norm",          grad_norm.item(),         step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "epoch_progress",     epoch + batch_idx / total_batches, step=global_step)
                    # GPU memory tracking
                    mlflow_client.log_metric(mlflow_run_id, "gpu_memory_allocated_gb",
                                             torch.cuda.memory_allocated() / 1024**3, step=global_step)
                    mlflow_client.log_metric(mlflow_run_id, "gpu_memory_reserved_gb",
                                             torch.cuda.memory_reserved() / 1024**3,  step=global_step)

                # Validation + checkpoint every CKPT_EVERY steps
                if global_step > 0 and global_step % CKPT_EVERY == 0:
                    val_metrics = run_validation(model, val_loader, device, max_batches=20)

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

            # End-of-epoch
            val_metrics   = run_validation(model, val_loader, device, max_batches=50)
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
# Driver
# ══════════════════════════════════════════════════════════════════════════════

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

    # ── Resume config ─────────────────────────────────────────────────────────
    # Set to the Ray experiment name to resume, or None for a fresh run.
    # RAY_EXPERIMENT_NAME = None
    RAY_EXPERIMENT_NAME = "llama31_8b_fsdp_4bac0716"

    # ── Auto-detect latest valid checkpoint ───────────────────────────────────
    RESUME_FROM_CHECKPOINT = None
    experiment_name        = f"llama31_8b_fsdp_{uuid.uuid4().hex[:8]}"

    if RAY_EXPERIMENT_NAME:
        storage_dir = f"/mnt/cluster_storage/{RAY_EXPERIMENT_NAME}"
        if os.path.isdir(storage_dir):
            all_ckpts   = sorted([
                os.path.join(storage_dir, d)
                for d in os.listdir(storage_dir)
                if d.startswith("checkpoint_") and os.path.isdir(os.path.join(storage_dir, d))
            ])
            # Only include checkpoints with .metadata (incomplete = crashed mid-write)
            valid_ckpts = [c for c in all_ckpts if os.path.exists(os.path.join(c, ".metadata"))]
            incomplete  = set(all_ckpts) - set(valid_ckpts)
            if incomplete:
                logger.warning(f"Skipping {len(incomplete)} incomplete checkpoint(s):")
                for c in sorted(incomplete):
                    logger.warning(f"  {os.path.basename(c)}")
            if valid_ckpts:
                RESUME_FROM_CHECKPOINT = valid_ckpts[-1]
                experiment_name        = RAY_EXPERIMENT_NAME
                logger.info(f"[Resume] Latest valid checkpoint: {RESUME_FROM_CHECKPOINT}")
            else:
                logger.info(f"[Resume] No valid checkpoints in {storage_dir} — fresh start")
        else:
            logger.info(f"[Resume] {storage_dir} not found — fresh start")

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
                logger.info("[MLflow] run_meta.json has no run_id — creating new run")
        else:
            logger.info("[MLflow] No run_meta.json found — creating new run")

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
            "model":              "LLaMA-3.1-8B-Instruct",
        }.items():
            mlflow_client_driver.set_tag(mlflow_run_id, k, v)
        logger.info(f"[MLflow] Created new run: {mlflow_run_id}")

    print(f"MLflow run : {mlflow_run_id}")
    print(f"MLflow UI  : {MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{mlflow_run_id}")
    print()

    # ── Training config ───────────────────────────────────────────────────────
    train_loop_config = {
        "epochs":                 2,
        "learning_rate":          1e-5,
        "lr_min":                 1e-7,
        "batch_size":             1,        # 1 per GPU — LLaMA 8B + seq_len=512 fits ~20GB
        "seq_len":                SEQ_LEN,
        "cpu_offload":            False,     # set False if GPU has >40GB VRAM
        "adam_beta1":             0.9,
        "adam_beta2":             0.95,
        "weight_decay":           0.1,
        "adam_eps":               1e-8,
        "max_grad_norm":          1.0,      # gradient clipping
        "total_steps":            100 * 2, # update when changing epochs/dataset
        "mlflow_run_id":          mlflow_run_id,
        "resume_checkpoint_path": RESUME_FROM_CHECKPOINT,
    }

    scaling_config = ray.train.ScalingConfig(num_workers=5, use_gpu=True)

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

    print(f"Starting LLaMA-3.1-8B training | Ray experiment: {experiment_name}")
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model     = init_model()
    model.load_state_dict(torch.load(PATH_TO_FULL_MODEL, map_location="cpu"))
    model.eval()

    prompt = "The future of distributed AI training is"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(output[0], skip_special_tokens=True))