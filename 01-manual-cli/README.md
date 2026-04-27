# Manual Ray Cluster Setup with SSH

This guide walks you through setting up a **multi-node Ray cluster** using SSH to connect to head and worker machines.

## Overview

We'll:
- SSH into a **head node** and install/start Ray
- SSH into **worker nodes** and connect them to the head
- Use `ray start --head` on the head node
- Use `ray start --address='192.168.3.73:6379'` on workers
- Verify the cluster and run distributed tasks

**When to use this method:**
- Multi-machine deployments
- Learning how distributed clusters work
- Production bare-metal or cloud VM setups (< 100 nodes)
- On-premise infrastructure

## Prerequisites

### Required Machines

- **1 Head Node:** Dedicated machine to coordinate the cluster
  - Ubuntu 20.04+ (or compatible Linux)
  - 4GB RAM minimum
  - Static IP or hostname (e.g., `head-node.local` or `192.168.3.73`)
  
- **N Worker Nodes:** Machines to execute tasks
  - Ubuntu 20.04+ (same OS as head)
  - 4GB RAM minimum per node
  - Network connectivity to head node

### Required Access

- SSH access to all machines (with sudo privileges)
- Network connectivity between all nodes (check firewall rules)
- Port availability: `6379` (Ray client), `8265` (Dashboard), `10001` (GCS server)
- Python 3.8+ (or will be installed)

---

## Step 1: Prepare Your Head Node Machine

SSH into your head node:

```bash
# Connect to head node (192.168.3.73)
ssh -i <path-to-key.pem> ubuntu@192.168.3.73
```

Once connected, update system and install conda:

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y build-essential

# Install Miniconda (if not already installed)
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh

# Initialize conda
$HOME/miniconda/bin/conda init

# Reload your shell
source ~/.bashrc
```

Create a conda environment for Ray on the head node:

```bash
# Create conda environment named ray-env with Python 3.9
conda create -n ray-env python=3.9 -y

# Activate the environment
conda activate ray-env

# Upgrade pip
pip install --upgrade pip

# Install Ray with all libraries
pip install ray[default]

# Verify installation
python -c "import ray; print(f'Ray {ray.__version__} installed')"
```

Your head node IP is **192.168.3.73** - you'll use this to connect worker nodes.

---

## Step 2: Start the Ray Head Node

Still on the head node, start Ray:

```bash
# Ensure conda environment is activated
conda activate ray-env

# Start the head node
ray start --head --port=6379
```

Expected output:
```
----
Started Ray with:
  Local node IP: 192.168.3.73
  Dashboard available at http://192.168.3.73:8265
  Ray logs: /home/ubuntu/ray_results/...
----
```

**Keep this terminal open!** The head node must stay running.

Verify it's working:

```bash
# In the same terminal (or a new SSH session to head):
ray status
```

Expected output:
```
======== Cluster Stats ========
Node Count: 1
Available Resources: {'CPU': 4.0}
...
```

---

## Step 3: Prepare Worker Nodes

Open **new SSH sessions** to each worker node and repeat the setup:

```bash
# SSH into worker 1
ssh -i <path-to-key.pem> ubuntu@WORKER_NODE_1_IP

# Once connected:
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential

# Install Miniconda if not already installed
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
$HOME/miniconda/bin/conda init
source ~/.bashrc

# Create conda environment
conda create -n ray-env python=3.9 -y
conda activate ray-env

# Install Ray
pip install --upgrade pip
pip install ray[default]

# Verify installation
python -c "import ray; print(f'Ray {ray.__version__} installed')"
```

**Repeat this for each worker node** (worker 2, worker 3, etc.)

---

## Step 4: Connect Worker Nodes to Head Node

On **each worker node**, connect to the head:

```bash
# Ensure conda environment is activated on the worker
conda activate ray-env

# Connect to head node (192.168.3.73)
ray start --address='192.168.3.73:6379'
```

Expected output:
```
----
Started Ray with:
  Remote node IP: 192.168.3.71
  Connected to head at: 192.168.3.73:6379
----
```

Keep this running on each worker node!

---

## Step 5: Verify Your Multi-Node Cluster

Go back to your **head node** and check the cluster:

```bash
# On the head node (in any terminal)
ray status
```

Expected output with 1 head + 2 workers:
```
======== Cluster Stats ========
Node Count: 3
Total Available Resources: {'CPU': 12.0}

Node: ray-head (192.168.3.73)
  Used Resources: {}
  Available Resources: {'CPU': 4.0}

Node: ray-worker-1 (192.168.3.71)
  Used Resources: {}
  Available Resources: {'CPU': 4.0}

Node: ray-worker-2 (192.168.3.72)
  Used Resources: {}
  Available Resources: {'CPU': 4.0}
```

**Excellent!** All nodes connected! ✅

---

## Step 6: Run Example Jobs on the Cluster

On your **local machine** (or any machine with network access to head), create a Python script:

```bash
# On your local machine
cat > distributed_job.py << 'EOF'
import ray
import socket

# Connect to the remote Ray head node (192.168.3.73)
ray.init(address="ray://192.168.3.73:10001")

print(f"Connected to Ray cluster!")
print(f"Available resources: {ray.available_resources()}")

# Define a remote function (will execute on workers)
@ray.remote
def task_on_worker(task_id):
    """This runs on a worker node"""
    hostname = socket.gethostname()
    return f"Task {task_id} executed on {hostname}"

# Submit 6 tasks to run in parallel
print("\nSubmitting 6 tasks...")
futures = [task_on_worker.remote(i) for i in range(1, 7)]

# Collect results
results = ray.get(futures)

print("Results:")
for result in results:
    print(f"  {result}")

ray.shutdown()
print("\nDone!")
EOF
```

Run the job:

```bash
# Make sure you can reach the head node from your machine
# (network/firewall must allow port 10001)

pip install ray[default]  # If not already installed locally
python distributed_job.py
```

---

## Step 7: Monitor with Ray Dashboard

The Ray Dashboard provides real-time monitoring:

```bash
# On your local machine, open browser
http://192.168.3.73:8265
```

You'll see:
- **Cluster Overview:** All 3 nodes with resources
- **Jobs:** Your running jobs and tasks
- **Task Timeline:** Where each task ran
- **Worker Information:** CPU/memory per node

---

## Step 8: Stop the Cluster

When done, shutdown cleanly:

```bash
# On EACH worker node, press Ctrl+C in the terminal running:
# ray start --address='...'
# OR execute:
ray stop

# On the head node, press Ctrl+C in the terminal running:
# ray start --head
# OR execute:
ray stop
```

This gracefully shuts down all Ray processes.

---

## Network Connectivity Checklist

If nodes can't connect, verify:

```bash
# From worker node, ping the head (192.168.3.73):
ping 192.168.3.73

# From worker node, test port connectivity:
telnet 192.168.3.73 6379

# Check firewall (may need to open ports):
sudo ufw allow 6379
sudo ufw allow 8265
sudo ufw allow 10001

# Check if head node is listening:
netstat -ln | grep 6379
```

---

## Common Configuration Options

### Head Node Setup with Conda (Quick Reference)

```bash
# SSH to head node
ssh ubuntu@192.168.3.73

# Install Miniconda
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda && rm miniconda.sh
$HOME/miniconda/bin/conda init && source ~/.bashrc

# Create and activate ray-env
conda create -n ray-env python=3.9 -y
conda activate ray-env
pip install ray[default]

# Start head node
ray start --head --port=6379
```

### Worker Node Setup with Conda (Quick Reference)

Repeat this on each worker node (192.168.3.71, 192.168.3.72, 192.168.3.73, etc.):

```bash
# SSH to worker node (example: 192.168.3.71)
ssh ubuntu@192.168.3.71

# Install Miniconda
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda && rm miniconda.sh
$HOME/miniconda/bin/conda init && source ~/.bashrc

# Create and activate ray-env
conda create -n ray-env python=3.9 -y
conda activate ray-env
pip install ray[default]

# Connect to head (192.168.3.73)
ray start --address='192.168.3.73:6379'
```

---

## Advanced Ray Configuration

When starting Ray, you can customize resource allocation:

```bash
# Start head node with custom CPU and memory limits
ray start --head \
  --num-cpus=4 \
  --object-store-memory=2000000000  # 2GB
```

When connecting workers, specify resources:

```bash
# Start worker with specific CPU allocation
ray start --address='192.168.3.73:6379' \
  --num-cpus=2 \
  --num-gpus=1
```

---

## Troubleshooting Multi-Node Setup
```

Then connect to the head node:

```bash
# Connect to head node (192.168.3.73)
ray start --address='192.168.3.73:6379'
```

### Verify the Multi-Node Cluster

On the head node, check all connected nodes:

```bash
ray status
```

Expected output with 2 worker nodes:
```
======== Cluster Stats ========
Node Count: 3
Total Available Resources: {'CPU': 12.0, 'memory': 12000000000.0}
...
```

---

## Common Issues & Solutions

### Issue: "Port 6379 already in use"

```bash
# Start Ray on a different port
ray start --head --port=6380
```

### Issue: "Cannot find Python 3.9"

```bash
# Check available Python versions
python3 --version
which python3

# Install if missing
sudo apt install python3.9
```

### Issue: "Ray installation failed"

```bash
# Upgrade pip first
pip install --upgrade pip

# Try installation again
pip install ray[default]
```

### Issue: Worker can't connect to head node

```bash
# Verify network connectivity from worker to head (192.168.3.73)
ping 192.168.3.73

# Ensure firewall allows port 6379
sudo ufw allow 6379

# Check if head node is listening
netstat -ln | grep 6379
```

---

## Customizing Ray Configuration

When starting Ray, you can customize resource allocation:

```bash
# Start head node with custom CPU and memory limits
ray start --head \
  --num-cpus=4 \
  --object-store-memory=2000000000  # 2GB
```

Or, use environment variables before one startup:

```bash
# Set a 4GB object store
export RAY_OBJECT_STORE_MEMORY=4000000000

# Start Ray
ray start --head
```

---

## Remote Connections (Advanced)

If you want to run code on your laptop that connects to a remote Ray cluster:

```python
import ray

# Connect to remote head node (must have port 10001 open)
ray.init(address="ray://REMOTE_IP:10001")

@ray.remote
def remote_task(x):
    return x * 2

# Run on the remote cluster
result = ray.get(remote_task.remote(5))
print(result)  # Output: 10

ray.shutdown()
```

---

## Workshop Checklist

Before moving to advanced topics, confirm you can:

- [ ] Install Miniconda/Anaconda 
- [ ] Create conda environment: `conda create -n ray-env python=3.9`
- [ ] Activate conda environment: `conda activate ray-env`
- [ ] Install Ray and verify the version: `pip install ray[default]`
- [ ] SSH to head node and start Ray: `ray start --head`
- [ ] SSH to worker nodes and connect: `ray start --address='192.168.3.73:6379'`
- [ ] See the Ray Dashboard at `http://192.168.3.73:8265`
- [ ] Submit tasks using `@ray.remote`
- [ ] Retrieve results with `ray.get()`
- [ ] Stop the cluster cleanly: `ray stop`

---

## Resources

- [Ray Official Documentation](https://docs.ray.io/)
- [Ray GitHub Repository](https://github.com/ray-project/ray)
- [Ray Cluster Documentation](https://docs.ray.io/en/latest/cluster/getting-started.html)

