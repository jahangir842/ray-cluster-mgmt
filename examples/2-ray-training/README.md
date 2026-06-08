# Ray Train Examples

Distributed PyTorch training using `ray.train.torch.TorchTrainer` across the
8-node cluster. Scripts progress from a single-machine baseline through DDP and
FSDP2 to large language models.

## Scripts

| # | File | Model | Strategy |
|---|------|-------|----------|
| 1 | `1-train-single-machine-resnet152.py` | ResNet-152 | Single GPU baseline |
| 2 | `2-train-cluster-resnet152.py` | ResNet-152 | DDP across all GPUs |
| 3 | `3.train-cluster-resnet18-profiler.py` | ResNet-18 | DDP + TensorBoard profiler |
| 4 | `4-train-cluster-fsdp-ViT-small.py` | ViT-Small | FSDP2 |
| 5 | `5-train-cluster-fsdp-vit-H14.py` | ViT-H/14 | FSDP2 |
| 6 | `6-train-cluster-fsdp-llama3.2-8b.py` | Llama 3.1-8B | FSDP2 |
| 7 | `7-train-cluster-fsdp-llama3.2-8b-with-logs.py` | Llama 3.1-8B | FSDP2 + NCCL debug logs |
| 8 | `8-train-cluster-fsdp-gpt-2-wikitext2.py` | GPT-2 | FSDP2 on WikiText-2 |
| 9 | `9-train-cluster-fsdp-gpt-2-tiny-stories.py` | GPT-2 | FSDP2 on TinyStories |
| 10 | `10-train-cluster-fsdp-gpt-2-mlflow.py` | GPT-2 | FSDP2 + MLflow tracking |
| 11 | `11-train_llama2_13b_mlflow.py` | Llama 2-13B | FSDP2 + MLflow tracking |

## Run

```bash
# Single-machine baseline (no Ray needed)
python 1-train-single-machine-resnet152.py

# Cluster job (any script 2–11)
ray job submit --address="http://192.168.3.73:8265" --working-dir . \
  -- python 2-train-cluster-resnet152.py
```

## Requirements

- Ray cluster running — see [`installation/01-manual-cli/`](../../installation/01-manual-cli/README.md)
- NFS shared storage mounted at `/mnt/cluster_storage/` for checkpoints — see [`installation/05-shared-storage/`](../../installation/05-shared-storage/README.md)
- MLflow server at `http://192.168.3.73:5000` for scripts 10 and 11

```bash
pip install torch torchvision transformers datasets mlflow tensorboard
```

## Key patterns

**DDP (scripts 2–3):** `ray.train.torch.prepare_model()` + `prepare_data_loader()`

**FSDP2 (scripts 4–11):** `torch.distributed.fsdp.fully_shard()` with a device
mesh; models are loaded on `"meta"` device then moved to GPU to avoid OOM during
init of large models.

**Ray Train V2** is enabled in all cluster scripts:
```python
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
```
