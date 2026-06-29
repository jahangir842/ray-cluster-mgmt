# Multi-node vLLM-on-Ray serving, in Docker

Serves a large model (e.g. DeepSeek-R1-70B) with **pipeline parallelism across all 8
nodes**, using one **identical container image on every node** so the per-node
environment drift that plagues the bare-metal conda setup becomes impossible.

## Why Docker fixes the drift

Every bring-up failure on the bare-metal cluster traced back to *8 independently-built
conda envs* drifting apart:

| Bare-metal fault | Root cause | Under Docker |
|---|---|---|
| `.74` torch 2.11 / vllm 0.23 (`ncclCommWindowDeregister`) | per-node pip drift | image is immutable |
| `.72` triton 3.2 vs torch 2.5.1 (`AttrsDescriptor` TypeError) | half-applied upgrade | can't half-apply an image |
| `.72` env at `~/.conda` vs others at `~/anaconda3` | per-node install paths | one path, baked in |
| `.71` duplicate / `.76` dead raylet | manual `ray start` cruft | `cluster-down && cluster-up` = clean membership |

One image tag = one torch/triton/nccl/vllm/ray on all 8 nodes, by construction.

## Cluster facts (baked into the scripts)

- **Head:** `192.168.3.73`  · **Workers:** `.71 .72 .74 .75 .76 .77 .78`  · SSH user `user`
- **GPUs:** `.71–.75` RTX 4500 Ada (NIC `enp0s31f6`), `.76–.78` RTX 3090 (NIC `eno1`), all 24 GB.
  `node-up.sh` **auto-detects** each node's 192.168.3.x NIC, so NCCL/GLOO pin the right one — no hardcoding.
- **Weights:** on NFS at `/mnt/cluster_storage/models/DeepSeek-R1-70B`, mounted read-only into every container at `/models`.
- **Image:** `vllm/vllm-openai:v0.6.6` (matches the bare-metal vLLM version).

## Files

| File | Runs on | Purpose |
|---|---|---|
| `node-up.sh` | each node (via NFS) | start one Ray container (head/worker) with the right NIC, GPU, shm, weights mount |
| `cluster-up.sh` | head | publish `node-up.sh` to NFS, start head + workers, wait for 8 GPUs, launch `vllm serve` |
| `cluster-down.sh` | head | remove the container on all nodes |
| `stop-baremetal-ray.sh` | head | stop the conda Ray cluster first (frees port 6379) |

## Usage

```bash
# on the head (192.168.3.73):
cd ~/projects/ray-cluster-mgmt/examples/6.vllm/docker-multinode

./stop-baremetal-ray.sh     # one-time: free 6379 from the bare-metal cluster
./cluster-up.sh             # head + 7 workers + vllm serve, PP=8

docker exec vllm-ray tail -f /tmp/vllm.log     # watch 70B load across 8 nodes
curl http://192.168.3.73:8000/v1/models        # ready when this returns the model
```

Tear down with `./cluster-down.sh`. To go back to bare-metal, run
`ray start --head ...` on the head and `ray start --address=192.168.3.73:6379` on each worker.

## The three settings that make multi-node Ray+NCCL work in Docker

Baked into `node-up.sh`; listed here because they're the usual failure points:

1. **`--network host`** — Ray uses dozens of dynamic ports (GCS, object manager, worker
   range 10002–19999) and so does NCCL. Bridge networking with port maps does not work for multi-node.
2. **`--shm-size=16g`** — Ray's plasma object store and NCCL use `/dev/shm`; the 64 MB
   Docker default crashes immediately.
3. **`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`** — pinned to each node's detected
   192.168.3.x NIC (`enp0s31f6` or `eno1`) so collectives don't wander onto `docker0`/virtual interfaces.

Plus `VLLM_HOST_IP` per node, so PP placement sees 8 distinct IPs.

## Memory / sizing note

70B FP16 ≈ 141 GB ÷ 8 stages ≈ 17.6 GB/stage on 24 GB cards — fits with room for KV cache
at `--gpu-memory-utilization 0.92`. PP=6 would be ~23.5 GB/stage and OOM, so keep PP=8.
For longer context or headroom, use an FP8/AWQ checkpoint.
