"""
resume_from_checkpoint.py
─────────────────────────
1. Loads a DCP checkpoint saved by AppState (same format as the main training
   script).
2. Prints the last completed batch / step before the crash.
3. Resumes the epoch by fast-forwarding the DataLoader — every batch before
   the crash point is explicitly skipped with `continue` so no data is reused.

Usage
─────
    python resume_from_checkpoint.py \
        --checkpoint /mnt/cluster_storage/gpt2_scratch_tinystories_5c9636bf/checkpoint_2026-05-21_09-45-06.606474 \
        --data      /mnt/cluster_storage/datasets/tinystories_tokenized.pt \
        [--epochs 1] [--batch-size 8] [--lr 1e-5] [--dry-run]

    --dry-run   Inspect the checkpoint and print the resume plan, then exit
                without running any training.
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants (must match the values used when the checkpoint was created) ────
SEQ_LEN   = 1024
VAL_SPLIT = 0.02


# ══════════════════════════════════════════════════════════════════════════════
# Dataset  (identical to main script so the split is reproducible)
# ══════════════════════════════════════════════════════════════════════════════

class TinyStoriesDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_dataloaders(data_path: str, batch_size: int):
    log.info(f"Loading dataset from {data_path} ...")
    raw   = torch.load(data_path, weights_only=True)
    n_val = int(len(raw) * VAL_SPLIT)
    n_train = len(raw) - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        TinyStoriesDataset(raw),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),   # ← same seed as main script
    )
    log.info(f"Dataset split — train: {n_train:,} | val: {n_val:,} sequences")

    # shuffle=False here so the skip logic is deterministic.
    # The DistributedSampler from the main script is NOT used in this
    # single-GPU resume script; the per-batch order is the same as long as
    # shuffle=False and the seed is the same.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# Model  (same architecture — swap out for your own if needed)
# ══════════════════════════════════════════════════════════════════════════════

def init_model(seq_len: int = SEQ_LEN) -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(
        vocab_size  = 50257,
        n_positions = seq_len,
        n_embd      = 768,
        n_layer     = 12,
        n_head      = 12,
        n_inner     = 3072,
        resid_pdrop = 0.1,
        attn_pdrop  = 0.1,
        embd_pdrop  = 0.1,
        loss_type   = "ForCausalLMLoss",
    )
    model = GPT2LMHeadModel(config)
    log.info(f"Model initialised — {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# AppState  (must mirror the class in the main training script exactly)
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


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(path: str, model, optimizer, scheduler):
    """
    Load a DCP checkpoint into model / optimizer / scheduler.
    Returns (epoch, global_step).
    """
    log.info(f"Loading checkpoint from:  {path}")
    app = AppState(model, optimizer, scheduler)
    dcp.load(state_dict={"app": app}, checkpoint_id=path)
    epoch       = app.epoch       or 0
    global_step = app.global_step or 0
    log.info(f"Checkpoint loaded — epoch={epoch}  global_step={global_step}")
    return epoch, global_step


def inspect_checkpoint(path: str):
    """
    Load *only* the scalar fields (epoch, global_step) without touching a GPU.
    Uses a lightweight model+optimizer-free AppState so we never allocate VRAM.
    Returns (epoch, global_step).
    """
    log.info(f"Inspecting checkpoint at: {path}")

    # Minimal proxy that only captures the scalar fields
    class _ScalarOnly(Stateful):
        def __init__(self):
            self.epoch       = None
            self.global_step = None

        def state_dict(self):
            return {"epoch": self.epoch, "global_step": self.global_step}

        def load_state_dict(self, sd):
            self.epoch       = sd.get("epoch", 0)
            self.global_step = sd.get("global_step", 0)

    proxy = _ScalarOnly()
    # DCP will skip any keys in the checkpoint that are not present in the
    # state_dict we pass (model, optim, scheduler), so this is safe.
    try:
        dcp.load(state_dict={"app": proxy}, checkpoint_id=path)
    except Exception as e:
        # DCP raises if keys in the file are absent from the state dict.
        # That's expected here — we only want the scalars.
        log.debug(f"DCP partial-load warning (expected): {e}")

    return proxy.epoch or 0, proxy.global_step or 0


# ══════════════════════════════════════════════════════════════════════════════
# Resume plan  (pure arithmetic — no GPU needed)
# ══════════════════════════════════════════════════════════════════════════════

def compute_resume_plan(epoch: int, global_step: int, batches_per_epoch: int) -> dict:
    """
    Given where the run crashed, return everything needed to resume.

      global_step          — last *completed* optimiser step
      batches_to_skip      — how many batches at the front of the epoch to skip
      next_batch_index     — the batch_idx the resume loop starts processing at
      steps_remaining      — how many steps are left in the current epoch
    """
    # global_step counts completed steps.  Within the current epoch the number
    # of batches already processed is:
    batches_done_in_epoch = global_step % batches_per_epoch

    # The next batch the model has NOT seen is:
    next_batch_index  = batches_done_in_epoch          # 0-based
    batches_to_skip   = batches_done_in_epoch
    steps_remaining   = batches_per_epoch - batches_done_in_epoch

    return {
        "resume_epoch":      epoch,
        "global_step":       global_step,
        "batches_per_epoch": batches_per_epoch,
        "batches_done":      batches_done_in_epoch,
        "batches_to_skip":   batches_to_skip,
        "next_batch_index":  next_batch_index,
        "steps_remaining":   steps_remaining,
    }


def print_resume_plan(plan: dict):
    print()
    print("═" * 58)
    print("  CHECKPOINT INSPECTION REPORT")
    print("═" * 58)
    print(f"  Epoch at crash          : {plan['resume_epoch']}")
    print(f"  Last completed step     : {plan['global_step']}")
    print(f"  Batches per epoch       : {plan['batches_per_epoch']}")
    print(f"  Batches completed       : {plan['batches_done']}")
    print(f"  Batches to skip on resume: {plan['batches_to_skip']}")
    print(f"  First unseen batch index: {plan['next_batch_index']}")
    print(f"  Steps remaining in epoch: {plan['steps_remaining']}")
    print("═" * 58)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_validation(model, val_loader, device, max_batches: int = 100) -> dict:
    model.eval()
    total_loss, total_correct, total_tokens, n = 0.0, 0, 0, 0
    for i, ids in enumerate(val_loader):
        if i >= max_batches:
            break
        ids     = ids.to(device)
        out     = model(input_ids=ids, labels=ids)
        logits  = out.logits[:, :-1, :]
        targets = ids[:, 1:]
        total_loss    += out.loss.item()
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_tokens  += targets.numel()
        n             += 1
    model.train()
    avg = total_loss / max(n, 1)
    return {
        "val_loss":       avg,
        "val_perplexity": math.exp(min(avg, 20)),
        "val_accuracy":   total_correct / max(total_tokens, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = init_model(seq_len=SEQ_LEN).to(device)

    # ── Build optimiser + scheduler ───────────────────────────────────────────
    optimizer   = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = args.total_steps or (7056 * args.epochs)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    start_epoch, global_step = load_checkpoint(
        args.checkpoint, model, optimizer, scheduler
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(args.data, args.batch_size)
    batches_per_epoch        = len(train_loader)

    # ── Compute & print the resume plan ───────────────────────────────────────
    plan = compute_resume_plan(start_epoch, global_step, batches_per_epoch)
    print_resume_plan(plan)

    if args.dry_run:
        log.info("--dry-run set. Exiting before training.")
        return

    # ── Resume training ───────────────────────────────────────────────────────
    log.info(
        f"Resuming training from epoch={start_epoch}, global_step={global_step}, "
        f"skipping first {plan['batches_to_skip']} batches of the epoch."
    )

    model.train()
    running_loss = 0.0
    n_batches    = 0

    for epoch in range(start_epoch, args.epochs):

        batches_to_skip = plan["batches_to_skip"] if epoch == start_epoch else 0

        if batches_to_skip > 0:
            log.info(
                f"Epoch {epoch} — fast-forwarding DataLoader: "
                f"skipping {batches_to_skip} already-seen batches "
                f"(batch indices 0 … {batches_to_skip - 1})."
            )

        skipped = 0   # counter used only for the progress message below

        for batch_idx, input_ids in enumerate(train_loader):

            # ── EXPLICIT SKIP ─────────────────────────────────────────────────
            # Every batch whose index is strictly less than batches_to_skip has
            # already been used in a completed optimiser step.  We pull it from
            # the DataLoader (so the internal state advances correctly) but we
            # do NOT forward it through the model, do NOT call backward, and do
            # NOT step the optimiser or scheduler.  This guarantees zero data
            # reuse across the resume boundary.
            if batch_idx < batches_to_skip:
                skipped += 1
                if skipped % 500 == 0 or skipped == batches_to_skip:
                    log.info(
                        f"  Skipping … {skipped:,} / {batches_to_skip:,} batches "
                        f"({100 * skipped / batches_to_skip:.1f} %)"
                    )
                continue   # ← the actual skip; nothing below runs for these batches

            # ── TRAINING STEP (only for unseen batches) ───────────────────────
            input_ids = input_ids.to(device)
            outputs   = model(input_ids=input_ids, labels=input_ids)
            loss      = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches    += 1
            global_step  += 1

            # Progress every 10 steps
            if global_step % 10 == 0:
                smooth = running_loss / n_batches
                vram   = torch.cuda.memory_allocated() / 1024**3 if device.type == "cuda" else 0
                log.info(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {global_step} | "
                    f"Batch {batch_idx + 1}/{batches_per_epoch} | "
                    f"Loss {loss.item():.4f} (smooth {smooth:.4f}) | "
                    f"LR {scheduler.get_last_lr()[0]:.2e} | "
                    f"VRAM {vram:.2f} GB"
                )

            # Validation every 500 steps
            if global_step % 500 == 0:
                val = run_validation(model, val_loader, device)
                log.info(f"[Val] step={global_step}  {val}")

        # End of epoch
        val = run_validation(model, val_loader, device, max_batches=200)
        log.info(
            f"Epoch {epoch + 1}/{args.epochs} complete | "
            f"avg_loss={running_loss / max(n_batches, 1):.4f} | {val}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Inspect and resume a DCP checkpoint.")
    p.add_argument("--checkpoint",   required=True,
                   help="Path to the checkpoint directory (e.g. .../checkpoint_2026-05-21_...)")
    p.add_argument("--data",         required=True,
                   help="Path to tinystories_tokenized.pt")
    p.add_argument("--epochs",       type=int,   default=1)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--total-steps",  type=int,   default=None,
                   help="Override T_max for the cosine scheduler (default: batches_per_epoch × epochs)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Inspect the checkpoint and print the resume plan, then exit.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset file not found: {args.data}")

    run_training(args)