# Manual Ray Cluster Setup with `ray start`

This guide walks you through setting up a Ray cluster manually using the `ray start` command on bare-metal or cloud instances.

## Overview

The `ray start` command is the simplest way to get Ray running. It:
- Installs Ray and dependencies on a single Python environment
- Starts the Ray head node (central coordinator)
- Optionally connects worker nodes to the head node

**When to use this method:**
- Learning and prototyping
- Fixed, small clusters (< 10 nodes)
- Bare-metal servers or IaaS instances (EC2, GCP VMs, etc.)
- Quick local testing

## Prerequisites

- Python 3.8 or later
- Ubuntu 20.04+ (or compatible Linux distribution)
- Basic knowledge of SSH and command-line tools
- 4GB RAM minimum per node

## Step 1: Update System and Install Python (5 minutes)

Execute these commands one by one to understand each step:

```bash
# Update package manager lists
sudo apt update && sudo apt upgrade -y
```

This fetches the latest package list from Ubuntu repositories.

```bash
# Install Python, pip, and build tools
sudo apt install -y python3.9 python3.9-venv python3.9-dev python3-pip build-essential
```

Breaking this down:
- `python3.9` - Python interpreter
- `python3.9-venv` - Virtual environment support for Python 3.9
- `python3.9-dev` - Development headers (needed for compiling some packages)
- `python3-pip` - Package installer
- `build-essential` - C compiler and build tools

(Optional) Make Python 3.9 the default:
```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
```

## Step 2: Create and Activate Virtual Environment (2 minutes)

A virtual environment isolates Ray and its dependencies from system Python:

```bash
# Create a virtual environment in the current directory
python3 -m venv ray-env
```

This creates a folder called `ray-env/` with a complete isolated Python installation.

```bash
# Activate the virtual environment
source ray-env/bin/activate
```

After running this, your terminal prompt should change to show `(ray-env)` prefix. **All subsequent commands must be run with this environment activated.**

If you close the terminal, remember to re-activate:
```bash
source ray-env/bin/activate
```

```bash
# Upgrade pip (the package installer)
pip install --upgrade pip
```

Newer versions of pip have better dependency resolution, so it's good practice to upgrade.

## Step 3: Install Ray (3-5 minutes)

With your virtual environment activated, install Ray:

```bash
# Install Ray with default libraries (Core, Tune, Train, Serve, Data)
pip install ray[default]
```

The `[default]` extra includes:
- **ray** - Core distributed computing framework
- **ray[tune]** - Hyperparameter tuning library
- **ray[train]** - Distributed training library
- **ray[serve]** - Model serving library
- **ray[data]** - Data processing library

If you want only the core Ray (smaller download):
```bash
pip install ray
```

Verify the installation:

```bash
# Print Ray version
python3 -c "import ray; print(f'Ray {ray.__version__} installed successfully!')"
```

You should see output like: `Ray 2.0.1 installed successfully!`

## Step 4: Start the Ray Head Node (1 minute)

Now you'll start the Ray head node, which is the central coordinator for your cluster:

```bash
# Start Ray with the head node role
ray start --head --port=6379
```

Let's break down what happens:
- `ray start` - Starts a Ray daemon
- `--head` - This is the head/master node
- `--port=6379` - Listen on port 6379 (default Ray client port)

Expected output:
```
----
Started Ray with:
  Local node IP: 127.0.0.1
  Dashboard available at 127.0.0.1:8265
  Experiment logs will be logged to /home/user/ray_results/
----
```

What just happened:
- ✅ Ray head node is now running
- ✅ Object store (in-memory data storage) initialized
- ✅ Task scheduler ready to receive tasks
- ✅ Dashboard accessible at `http://127.0.0.1:8265`

To see cluster status anytime:

```bash
# Check Ray cluster status
ray status
```

This shows nodes, available CPU/GPU, memory, and other resources.

## Step 5: Run Your First Ray Job (2 minutes)

Keep the head node running (don't close that terminal). **Open a new terminal** and create a Python script:

```bash
# Create the example job file
cat > example-job.py << 'EOF'
import ray
import time

print("Connecting to Ray cluster...")
ray.init(ignore_reinit_error=True)

print("Ray cluster info:")
print(f"  Jobs: {ray.available_resources()}")

# Define a remote function (will run on workers)
@ray.remote
def expensive_task(x):
    """This function runs on the Ray cluster"""
    time.sleep(1)  # Simulate work
    return x * x

print("\nSubmitting 5 tasks to the cluster...")
# Submit tasks for parallel execution
futures = [expensive_task.remote(i) for i in range(1, 6)]

print("Waiting for results...")
# Collect results (blocks until all tasks complete)
results = ray.get(futures)

print(f"\nResults from parallel tasks:")
for i, result in enumerate(results, 1):
    print(f"  Task {i}: {i}² = {result}")

print("\nShutting down Ray...")
ray.shutdown()
print("Done!")
EOF
```

Now run the job:

```bash
# Make sure your virtual environment is activated first!
source ray-env/bin/activate

# Run the example job
python example-job.py
```

You should see:
```
Connecting to Ray cluster...
Ray cluster info:
  Jobs: {'CPU': 4.0}

Submitting 5 tasks to the cluster...
Waiting for results...

Results from parallel tasks:
  Task 1: 1² = 1
  Task 2: 2² = 4
  Task 3: 3² = 9
  Task 4: 4² = 16
  Task 5: 5² = 25

Shutting down Ray...
Done!
```

**What you just learned:**
- `@ray.remote` decorator turns a function into a distributed task
- `.remote()` submits the task to the cluster (non-blocking)
- `ray.get()` waits for results and retrieves them
- Ray automatically handles scheduling and execution

## Step 6: Explore the Ray Dashboard (2 minutes)

The Ray Dashboard is a web UI to monitor your cluster in real-time:

```bash
# Open your browser (or click the URL from terminal output)
http://localhost:8265
```

In the Dashboard, you'll see:
- **Cluster** tab: CPU, GPU, memory per node
- **Jobs** tab: Your running/completed jobs
- **Actors** tab: Long-running stateful processes
- **Logs** tab: System and application logs
- **Metrics** tab: Performance graphs

This is invaluable for understanding what Ray is doing under the hood!

## Step 7: Stop the Ray Cluster

When you're done, stop Ray cleanly:

```bash
# In the terminal where Ray is still running (Ctrl+C)
# OR in a new terminal:
ray stop
```

This gracefully shuts down all Ray processes.

## Understanding What We Just Did

Let's break down the workshop flow:

1. **Environment Setup** (Step 1-3)
   - Created isolated Python environment
   - Installed Ray framework
   
2. **Started Head Node** (Step 4)
   - Central coordinator for the cluster
   - Manages task scheduling
   
3. **Submitted Tasks** (Step 5)
   - Defined remote functions
   - Submitted tasks via `.remote()`
   - Collected results via `ray.get()`
   
4. **Monitored** (Step 6)
   - Visualized cluster state
   - Saw task execution in real-time

---

## Adding Worker Nodes to Your Cluster (Advanced)

If you have **additional machines** on the same network, you can add them as worker nodes:

### On the Worker Machine:

First, prepare the environment (repeat Steps 1-3 on the worker machine):

```bash
# Update and install Python
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.9 python3.9-venv python3.9-dev python3-pip build-essential

# Create virtual environment
python3 -m venv ray-env
source ray-env/bin/activate

# Install Ray
pip install ray[default]
```

Then connect to the head node:

```bash
# Replace HEAD_IP with your head node's actual IP address
# Find it by running: hostname -I
ray start --address='HEAD_IP:6379'
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
# Verify network connectivity from worker to head
ping HEAD_IP

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

- [ ] Create a Python virtual environment
- [ ] Install Ray and verify the version
- [ ] Start a Ray head node
- [ ] See the Ray Dashboard at `http://localhost:8265`
- [ ] Submit tasks using `@ray.remote`
- [ ] Retrieve results with `ray.get()`
- [ ] Stop the cluster cleanly
- [ ] (Optional) Connect a worker node

---

## Resources

- [Ray Official Documentation](https://docs.ray.io/)
- [Ray GitHub Repository](https://github.com/ray-project/ray)
- [Ray Cluster Documentation](https://docs.ray.io/en/latest/cluster/getting-started.html)

