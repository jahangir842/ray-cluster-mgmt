import os
import tempfile
import uuid
import logging
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
from torch.utils.data import DataLoader, Dataset
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

# Enable Ray Train V2
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"

# ── MLflow config ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.3.73:5000")
MLFLOW_EXPERIMENT   = "gpt2-tinystories-2"

_NCCL_ENV = {
    "NCCL_SOCKET_IFNAME":      "enp0s31f6,eno1",
    "GLOO_SOCKET_IFNAME":      "enp0s31f6,eno1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "RAY_DEDUP_LOGS":          "0",
    "MLFLOW_TRACKING_URI":     MLFLOW_TRACKING_URI,
    "HF_HOME":                 "/mnt/cluster_storage/.cache/huggingface",
}
os.environ.update(_NCCL_ENV)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEQ_LEN        = 1024
TOKENIZER_PATH = "/mnt/cluster_storage/datasets/gpt2_tokenizer"
TOKENIZED_PATH = "/mnt/cluster_storage/datasets/tinystories_tokenized.pt"


# ── Dataset ───────────────────────────────────────────────────────────────────

class TinyStoriesDataset(Dataset):
    def __init__(self, seq_len: int):
        logger.info(f"Loading pre-tokenized dataset from {TOKENIZED_PATH} ...")
        self.data = torch.load(TOKENIZED_PATH)
        logger.info(f"Loaded {len(self.data):,} sequences of length {seq_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

def init_model() -> torch.nn.Module:
    from transformers import GPT2Config
    logger.info("Initializing blank GPT-2 from config (no pretrained weights)...")
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        loss_type="ForCausalLMLoss",
    )
    model = GPT2LMHeadModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Blank GPT-2 initialized — {total_params:,} parameters")
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
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    for decoder_block in model.transformer.h:
        fully_shard(decoder_block, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
    logger.info("FSDP2 sharding complete.")


# ── Checkpointing ─────────────────────────────────────────────────────────────

class AppState(Stateful):
    def __init__(self, model, optimizer=None, epoch=None):
        self.model     = model
        self.optimizer = optimizer
        self.epoch     = epoch

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        if "epoch" in state_dict:
            self.epoch = state_dict["epoch"]


def load_fsdp_checkpoint(model, optimizer, ckpt):
    logger.info("Loading distributed checkpoint for resuming training...")
    try:
        with ckpt.as_directory() as checkpoint_dir:
            app_state = AppState(model, optimizer)
            dcp.load(state_dict={"app": app_state}, checkpoint_id=checkpoint_dir)
        logger.info(f"Loaded checkpoint from epoch {app_state.epoch}")
        return app_state.epoch
    except Exception as e:
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e


def report_metrics_and_save_fsdp_checkpoint(
    model, optimizer, metrics,
    epoch=0, batch=0,
    is_rank0=False,
    mlflow_run_id=None,
    mlflow_client=None,
):
    """Save DCP checkpoint and log metrics to MLflow.

    IMPORTANT: Uses MlflowClient.log_metrics() directly — never
    `with mlflow.start_run()`. Context managers terminate the run
    when the with-block exits, marking it "Finished" mid-training.
    """
    logger.info("Saving checkpoint and reporting metrics...")

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        dcp.save(
            state_dict={"app": AppState(model, optimizer, epoch)},
            checkpoint_id=temp_checkpoint_dir,
        )
        ray.train.report(
            metrics,
            checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
        )

    if is_rank0 and mlflow_run_id and mlflow_client:
        global_step = epoch * 10_000 + batch
        mlflow_client.log_metrics(
            mlflow_run_id,
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            step=global_step,
        )
        logger.info(f"[MLflow] Logged metrics at step {global_step}: {metrics}")

    logger.info(f"Checkpoint saved. Metrics: {metrics}")


def save_model_for_inference(
    model, world_rank,
    mlflow_run_id=None, mlflow_client=None, experiment_name="",
):
    logger.info("Preparing model for inference — all-gathering shards to rank 0...")
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        save_file        = os.path.join(temp_checkpoint_dir, "full-model.pt")
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
            if mlflow_run_id and mlflow_client:
                mlflow_client.log_artifact(mlflow_run_id, save_file, artifact_path="full_model")
                logger.info("[MLflow] Logged full-model.pt as artifact")
        ray.train.report({}, checkpoint=checkpoint, checkpoint_dir_name="full_model")


# ── Training Function ─────────────────────────────────────────────────────────

def train_func(config):
    """Main training function — blank GPT-2 on TinyStories with FSDP2 + MLflow."""

    ctx        = ray.train.get_context()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    is_rank0   = (world_rank == 0)

    mlflow_run_id = config.get("mlflow_run_id")

    # ── MLflow client setup (rank 0 only) ─────────────────────────────────────
    # MlflowClient logs to an existing run without opening/closing it.
    # The run lifecycle (RUNNING -> FINISHED/FAILED) is managed only by
    # the driver process via set_terminated() after trainer.fit() returns.
    mlflow_client = None
    if is_rank0 and mlflow_run_id:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        mlflow_client.log_params(mlflow_run_id, {
            "epochs":        config.get("epochs", 2),
            "learning_rate": config.get("learning_rate", 1e-5),
            "batch_size":    config.get("batch_size", 8),
            "seq_len":       config.get("seq_len", SEQ_LEN),
            "world_size":    world_size,
            "model":         "gpt2-124M-scratch",
            "dataset":       "TinyStories",
        })
        logger.info(f"[MLflow] Client ready — run {mlflow_run_id} at {MLFLOW_TRACKING_URI}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = init_model()
    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    shard_model(model)

    optimizer = Adam(model.parameters(), lr=config.get("learning_rate", 1e-5))

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch       = 0
    loaded_checkpoint = ray.train.get_checkpoint()
    if loaded_checkpoint:
        latest_epoch = load_fsdp_checkpoint(model, optimizer, loaded_checkpoint)
        start_epoch  = latest_epoch + 1 if latest_epoch is not None else 0
        logger.info(f"Resuming from epoch {start_epoch}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_data   = TinyStoriesDataset(seq_len=config.get("seq_len", SEQ_LEN))
    train_loader = DataLoader(
        train_data,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
    )
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    epochs        = config.get("epochs", 2)
    total_batches = len(train_loader)

    logger.info(
        f"Training config — epochs: {epochs} | batches/epoch: {total_batches} | "
        f"batch_size: {config.get('batch_size', 4)} | "
        f"seq_len: {config.get('seq_len', SEQ_LEN)} | "
        f"world_size: {world_size}"
    )

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

        running_loss = 0.0
        num_batches  = 0

        for epoch in range(start_epoch, epochs):
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            for batch_idx, input_ids in enumerate(train_loader):
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss    = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prof.step()

                running_loss += loss.item()
                num_batches  += 1

                if batch_idx % 10 == 0:
                    vram = torch.cuda.memory_allocated() / 1024**3
                    logger.info(
                        f"[Rank {world_rank}] Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx+1}/{total_batches} | "
                        f"Loss: {loss.item():.4f} | VRAM: {vram:.2f} GB"
                    )

                # ── checkpoint every 500 batches ──────────────────────────────
                if batch_idx > 0 and batch_idx % 500 == 0:
                    mid_loss = running_loss / num_batches
                    logger.info(
                        f"[Rank {world_rank}] Mid-epoch checkpoint at batch {batch_idx} "
                        f"| Loss: {mid_loss:.4f}"
                    )
                    report_metrics_and_save_fsdp_checkpoint(
                        model, optimizer,
                        metrics={
                            "loss":       mid_loss,
                            "perplexity": torch.exp(torch.tensor(mid_loss)).item(),
                            "epoch":      float(epoch),
                            "batch":      float(batch_idx),
                        },
                        epoch=epoch, batch=batch_idx,
                        is_rank0=is_rank0,
                        mlflow_run_id=mlflow_run_id,
                        mlflow_client=mlflow_client,
                    )

            # ── end-of-epoch checkpoint ───────────────────────────────────────
            avg_loss = running_loss / num_batches
            metrics  = {
                "loss":       avg_loss,
                "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
            }
            report_metrics_and_save_fsdp_checkpoint(
                model, optimizer, metrics,
                epoch=epoch, batch=total_batches,
                is_rank0=is_rank0,
                mlflow_run_id=mlflow_run_id,
                mlflow_client=mlflow_client,
            )
            logger.info(f"Epoch {epoch+1}/{epochs} complete | {metrics}")

    # ── Export memory profiles and log to MLflow ──────────────────────────────
    run_name    = ctx.get_experiment_name()
    profile_dir = f"/mnt/cluster_storage/{run_name}"
    os.makedirs(profile_dir, exist_ok=True)
    profile_path = f"{profile_dir}/rank{world_rank}_memory_profile.html"
    prof.export_memory_timeline(profile_path)
    logger.info(f"[Rank {world_rank}] Memory profile saved to {profile_path}")

    if is_rank0 and mlflow_run_id and mlflow_client:
        for rank_id in range(world_size):
            rank_profile = f"{profile_dir}/rank{rank_id}_memory_profile.html"
            if os.path.exists(rank_profile):
                mlflow_client.log_artifact(
                    mlflow_run_id, rank_profile, artifact_path="memory_profiles"
                )
                logger.info(f"[MLflow] Logged memory profile for rank {rank_id}")

    # ── Save full model for inference ─────────────────────────────────────────
    save_model_for_inference(
        model, world_rank,
        mlflow_run_id=mlflow_run_id,
        mlflow_client=mlflow_client,
        experiment_name=run_name,
    )


# ── Launch Training ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env={"env_vars": _NCCL_ENV},
    )

    # ── Driver creates the run via MlflowClient and keeps it RUNNING ──────────
    # The driver never calls mlflow.start_run() — it uses the client API
    # exclusively. This means the run stays in RUNNING state until the driver
    # explicitly calls set_terminated() after training completes.
    mlflow_client_driver = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment = mlflow_client_driver.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        experiment_id = mlflow_client_driver.create_experiment(MLFLOW_EXPERIMENT)
        logger.info(f"Created MLflow experiment: {MLFLOW_EXPERIMENT}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment: {MLFLOW_EXPERIMENT} (id={experiment_id})")

    experiment_name = f"gpt2_scratch_tinystories_{uuid.uuid4().hex[:8]}"
    run             = mlflow_client_driver.create_run(
        experiment_id=experiment_id,
        run_name=experiment_name,
    )
    mlflow_run_id = run.info.run_id

    print(f"MLflow run created : {mlflow_run_id}")
    print(f"MLflow UI          : {MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{mlflow_run_id}")

    scaling_config = ray.train.ScalingConfig(num_workers=8, use_gpu=True)

    train_loop_config = {
        "epochs":        2,
        "learning_rate": 1e-5,
        "batch_size":    8,
        "seq_len":       1024,
        "mlflow_run_id": mlflow_run_id,
    }

    RESUME_FROM_CHECKPOINT = None
    # RESUME_FROM_CHECKPOINT = "/mnt/cluster_storage/gpt2_scratch_tinystories_.../checkpoint_..."

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

    print(f"Starting GPT-2 training | Ray experiment: {experiment_name}")
    print()

    try:
        result = trainer.fit()
        print("Training completed successfully!")
        # Driver is the sole owner of the run lifecycle
        mlflow_client_driver.set_terminated(mlflow_run_id, status="FINISHED")
        print(f"MLflow run marked FINISHED: {mlflow_run_id}")
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