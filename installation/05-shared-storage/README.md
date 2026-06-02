# Shared Storage Setup (NFS)

Ray Train checkpoints, MLflow artifacts, and TensorBoard traces all write to `/mnt/cluster_storage/`. This directory must be shared across **all nodes** so any worker can read what any other worker wrote.

This guide sets up an NFS export on the head node and mounts it on every worker.

---

## Architecture

```
Head Node (192.168.3.73)
  └── exports /mnt/cluster_storage  ──► NFS server

Worker Nodes (192.168.3.71 – .78)
  └── mounts 192.168.3.73:/mnt/cluster_storage → /mnt/cluster_storage
```

---

## Step 1: Head Node — Create and Export the Directory

SSH into the head node:

```bash
ssh user@192.168.3.73
```

Install the NFS server and create the storage directory:

```bash
sudo apt update
sudo apt install -y nfs-kernel-server

sudo mkdir -p /mnt/cluster_storage
sudo chmod 777 /mnt/cluster_storage
```

Add the export to `/etc/exports` (allows all cluster nodes to mount with full read/write):

```bash
sudo tee -a /etc/exports <<'EOF'
/mnt/cluster_storage 192.168.3.0/24(rw,sync,no_subtree_check,no_root_squash)
EOF
```

Apply the export and start the NFS service:

```bash
sudo exportfs -a
sudo systemctl enable --now nfs-kernel-server
sudo systemctl status nfs-kernel-server
```

Verify the export is published:

```bash
showmount -e 192.168.3.73
# Expected: /mnt/cluster_storage  192.168.3.0/24
```

---

## Step 2: Worker Nodes — Mount the Shared Directory

Repeat on **each worker node** (`.71`, `.72`, `.74`, `.75`, `.76`, `.77`, `.78`):

```bash
ssh user@192.168.3.71   # repeat for each worker IP
```

Install the NFS client and create the mount point:

```bash
sudo apt update
sudo apt install -y nfs-common

sudo mkdir -p /mnt/cluster_storage
```

Mount manually (to test):

```bash
sudo mount -t nfs 192.168.3.73:/mnt/cluster_storage /mnt/cluster_storage
```

Verify the mount:

```bash
df -h | grep cluster_storage
# Expected: 192.168.3.73:/mnt/cluster_storage  ...  /mnt/cluster_storage
```

---

## Step 3: Make Mounts Persistent (All Worker Nodes)

Add the mount to `/etc/fstab` so it survives reboots:

```bash
echo "192.168.3.73:/mnt/cluster_storage  /mnt/cluster_storage  nfs  defaults,_netdev  0  0" \
  | sudo tee -a /etc/fstab
```

Test that fstab is correct without rebooting:

```bash
sudo mount -a
df -h | grep cluster_storage
```

---

## Step 4: One-Shot Setup Across All Workers

Run this loop from the head node (requires passwordless SSH to all workers):

```bash
for ip in 192.168.3.71 192.168.3.72 192.168.3.74 192.168.3.75 192.168.3.76 192.168.3.77 192.168.3.78; do
  echo "=== $ip ==="
  ssh user@$ip "
    sudo apt install -y nfs-common &&
    sudo mkdir -p /mnt/cluster_storage &&
    sudo mount -t nfs 192.168.3.73:/mnt/cluster_storage /mnt/cluster_storage &&
    echo '192.168.3.73:/mnt/cluster_storage  /mnt/cluster_storage  nfs  defaults,_netdev  0  0' | sudo tee -a /etc/fstab &&
    echo 'Done'
  "
done
```

---

## Step 5: Smoke Test

Write a file from the head node and read it from a worker to confirm shared access:

```bash
# On head node
echo "shared storage works" > /mnt/cluster_storage/test.txt

# On any worker node
cat /mnt/cluster_storage/test.txt
# Expected: shared storage works
```

Clean up:

```bash
rm /mnt/cluster_storage/test.txt
```

---

## Directory Layout Used by This Cluster

```
/mnt/cluster_storage/
├── mlflow/
│   ├── mlflow.db           # MLflow experiment/run metadata
│   └── artifacts/          # Logged models, plots, files
├── checkpoints/            # Ray Train model checkpoints
└── tensorboard/            # TensorBoard profiler traces
```

These subdirectories are created automatically by the relevant services on first use.

---

## Firewall

If `ufw` is active on the head node, allow NFS traffic from the cluster subnet:

```bash
sudo ufw allow from 192.168.3.0/24 to any port nfs
sudo ufw reload
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|:--|:--|:--|
| `mount: connection timed out` | Firewall blocking NFS port | Open port 2049 on head node |
| `Permission denied` when writing | Wrong directory permissions | `sudo chmod 777 /mnt/cluster_storage` on head |
| Mount missing after reboot | fstab not updated | Re-run Step 3 |
| `rpcbind` errors | RPC service not running | `sudo systemctl start rpcbind` |
| Stale file handle | NFS server restarted without remounting | `sudo umount -f /mnt/cluster_storage && sudo mount -a` on worker |

---

## Next Steps

- [Start the Ray cluster](../01-manual-cli/README.md)
- [Set up MLflow tracking server](../04-mlflow/README.md)
