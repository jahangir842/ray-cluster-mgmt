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

# Model: LLaMA-3.1-8B-Instruct — 8B parameters, ~128 GB training footprint
# Single 24GB GPU cannot train this. Requires FSDP2 across multiple GPUs.
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

# Enable Ray Train V2
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"      # show all logs without deduplication

_NCCL_ENV = {
    "NCCL_SOCKET_IFNAME": "enp0s31f6,eno1",
    "GLOO_SOCKET_IFNAME": "enp0s31f6,eno1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "RAY_DEDUP_LOGS": "0",
    "NCCL_DEBUG": "INFO",            # ← verbose NCCL logs
    "NCCL_DEBUG_SUBSYS": "INIT,NET",
    "NCCL_IB_TIMEOUT": "30",
    "NCCL_TIMEOUT": "30",
}
os.environ.update(_NCCL_ENV)

# Set up logging — Ray Train workers write structured JSON logs to:
# /tmp/ray/session_latest/logs/train/ray-train-app-worker-*.log
# Watch live with:
#   LOG=$(ls -t /tmp/ray/session_latest/logs/train/ray-train-app-worker-*.log | head -1)
#   tail -f $LOG | python3 -c "
#     import sys, json
#     for line in sys.stdin:
#       try: d=json.loads(line); print(d['asctime'], f\"[rank {d.get('world_rank','?')}]\", d['message'])
#       except: print(line, end='')"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/home/user/projects/vllm-deployment/vllm/models/3.1-8b-instruct"
SEQ_LEN    = 512


# ── Dataset ───────────────────────────────────────────────────────────────────

class SyntheticTextDataset(Dataset):
    """Synthetic token dataset — no download needed.

    Generates random token IDs in LLaMA-3.1's vocabulary range (128256).
    Demonstrates FSDP2 sharding and distributed training infrastructure.
    Replace with a real dataset (WikiText, OpenWebText, etc.) for actual training.
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

    Memory stats:
        Parameters:      8,030,000,000
        Model size fp16: ~16.1 GB     (loaded directly in fp16)
        Adam states:     ~96.4 GB     (3x model in fp32, CPU-offloaded)
        Total training:  ~112 GB      → impossible on a single 24 GB GPU
        Per-GPU (8-way): ~14 GB       → fits with FSDP2 sharding

    Loaded from local path on every node — no internet download.
    """
    logger.info(f"Initializing LLaMA-3.1-8B-Instruct from {MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,      # fp16: 16 GB RAM per node vs 32 GB for fp32
        local_files_only=True,    # never hit the internet — use local copy
    )
    return model


# ── FSDP2 Sharding ────────────────────────────────────────────────────────────

def shard_model(model: torch.nn.Module):
    """Apply FSDP2 sharding to LLaMA-3.1-8B across all ranks.

    LLaMA transformer blocks live at model.model.layers.
    Each block is sharded independently (reshard_after_forward=True),
    then the outer model wrapper is sharded.

    With CPUOffloadPolicy, Adam optimizer states live in system RAM,
    not GPU VRAM — critical for fitting an 8B model on 24 GB cards.
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

    for decoder_block in model.model.layers:
        fully_shard(
            decoder_block,
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
    logger.info(f"Checkpoint saved. Metrics: {metrics}")


def save_model_for_inference(model: FSDPModule, world_rank: int) -> None:
    logger.info("Preparing model for inference — all-gathering shards to rank 0...")
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
    """Main training function — LLaMA-3.1-8B-Instruct with FSDP2 on Ray Train.

    Progress logs go to the structured JSON log file at:
        /tmp/ray/session_latest/logs/train/ray-train-app-worker-<id>.log

    Pretty-print live on any node with:
        LOG=$(ls -t /tmp/ray/session_latest/logs/train/ray-train-app-worker-*.log | head -1)
        tail -f $LOG | python3 -c "
          import sys, json
          for line in sys.stdin:
            try: d=json.loads(line); print(d['asctime'], f'[rank {d.get(chr(34)+'world_rank'+chr(34),chr(63))}]', d['message'])
            except: print(line, end='')"
    """
    model = init_model()

    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    # Do NOT call model.to(device) before sharding.
    # FSDP2 moves each rank's shard to GPU inside fully_shard().
    # Calling .to(device) first loads the full 16 GB onto one GPU → OOM.

    shard_model(model)

    optimizer = Adam(model.parameters(), lr=config.get("learning_rate", 1e-5))

    start_epoch = 0
    loaded_checkpoint = ray.train.get_checkpoint()
    if loaded_checkpoint:
        latest_epoch = load_fsdp_checkpoint(model, optimizer, loaded_checkpoint)
        start_epoch = latest_epoch + 1 if latest_epoch is not None else 0
        logger.info(f"Resuming from epoch {start_epoch}")

    # LLaMA-3.1 vocab size = 128256
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

    world_rank    = ray.train.get_context().get_world_rank()
    epochs        = config.get("epochs", 2)
    total_batches = len(train_loader)

    logger.info(
        f"Training config — "
        f"epochs: {epochs} | "
        f"batches/epoch: {total_batches} | "
        f"batch_size: {config.get('batch_size', 1)} | "
        f"seq_len: {config.get('seq_len', SEQ_LEN)} | "
        f"world_size: {ray.train.get_context().get_world_size()}"
    )

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
        num_batches  = 0

        for epoch in range(start_epoch, epochs):
            if ray.train.get_context().get_world_size() > 1:
                train_loader.sampler.set_epoch(epoch)

            for batch_idx, input_ids in enumerate(train_loader):
                # Causal LM — model computes cross-entropy loss internally
                # when labels == input_ids (next-token prediction objective)
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss    = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prof.step()

                running_loss += loss.item()
                num_batches  += 1

                # Progress log every 10 batches from every rank.
                # Writes to structured JSON log: logs/train/ray-train-app-worker-*.log
                if batch_idx % 10 == 0:
                    vram = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"[Rank {world_rank}] "
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx+1}/{total_batches} | "
                        f"Loss: {loss.item():.4f} | "
                        f"VRAM: {vram:.2f} GB"
                    )

            avg_loss = running_loss / num_batches
            metrics  = {
                "loss":       avg_loss,
                "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
            }
            report_metrics_and_save_fsdp_checkpoint(model, optimizer, metrics, epoch)
            logger.info(f"Epoch {epoch+1}/{epochs} complete | {metrics}")

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
        "batch_size":    1,     # batch=1 per GPU — 8B model needs headroom for NCCL buffers
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
    print(f"Experiment: {experiment_name}")
    print()
    print("Watch training progress live (run in another terminal on any node):")
    print("  LOG=$(ls -t /tmp/ray/session_latest/logs/train/ray-train-app-worker-*.log | head -1)")
    print('  tail -f $LOG | python3 -c "import sys,json; [print(json.loads(l)[\'asctime\'], json.loads(l)[\'message\']) if \'{\" in l else None for l in sys.stdin]"')
    print()

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