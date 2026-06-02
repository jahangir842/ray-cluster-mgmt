# Ray Cluster with Docker Compose

This guide shows how to deploy a multi-node Ray cluster using Docker Compose—perfect for local development, testing, and demo environments.

## Overview

Docker Compose lets you simulate a multi-node Ray cluster on a single machine using containers:
- **Head Node:** Central Ray coordinator
- **Worker Nodes:** Multiple worker containers connected to the head
- **Networking:** Automatic communication via Docker bridge network
- **Isolation:** Clean, reproducible environments

**When to use this method:**
- Local development and testing
- Demos and workshops
- CI/CD pipelines
- Cross-platform consistency (Windows, macOS, Linux)
- Avoiding dependency conflicts with `ray start`

## Prerequisites

- **Docker:** 20.10+ (install from [docker.com](https://www.docker.com/))
- **Docker Compose:** 1.29+ (often included with Docker Desktop)
- **4GB RAM** minimum (8GB+ recommended)
- **Disk Space:** 5GB for images and containers

### Verify Installation

```bash
docker --version
docker-compose --version
```

## Quick Start (3 minutes)

Execute these commands in order to bring up a 3-node Ray cluster:

```bash
# Step 1: Navigate to this directory
cd 02-docker-compose

# Step 2: Build the Docker image (one-time only)
docker-compose build
```

This creates a Docker image named `ray:latest` with:
- Python 3.9
- Ray[default] and common ML libraries
- A startup script that will be run when the container starts

```bash
# Step 3: Start the cluster (1 head + 2 workers)
docker-compose up -d
```

The `-d` flag runs in background. The `up` command will:
- Create and start the head container
- Wait for head to be healthy
- Create and start 2 worker containers
- Auto-connect workers to head

```bash
# Step 4: Check status
docker-compose logs -f ray-head
```

You should see output like:
```
ray-head | Started Ray with:
ray-head |   Dashboard available at 127.0.0.1:8265
```

```bash
# Step 5: View cluster info (in another terminal)
docker-compose exec ray-head ray status
```

Output:
```
======== Cluster Stats ========
Node Count: 3
Object Store Memory: 3.00 Gb
...
```

```bash
# Step 6: Run an example job
docker-compose exec ray-head python /app/example-job.py
```

You should see tasks executing on different worker nodes.

```bash
# Step 7: Stop the cluster when done
docker-compose down
```

---

```
02-docker-compose/
├── README.md                  # This file
├── docker-compose.yml         # Multi-container cluster definition
├── Dockerfile                 # Ray environment image
├── example-job.py             # Sample Ray job
└── .env                       # Environment variables (optional)
```

## Understanding the docker-compose.yml File

Before running Docker Compose, let's understand what's in the configuration file:

```yaml
version: '3.9'

services:
  ray-head:
    # Build from Dockerfile
    build:
      context: .
      dockerfile: Dockerfile
    
    # Container name for reference
    container_name: ray-head
    
    # Expose ports to host machine
    ports:
      - "6379:6379"      # Ray client port
      - "8265:8265"      # Ray dashboard
      - "10001:10001"    # Ray GCS server
    
    # Command to run when container starts
    command: >
      bash -c "
      ray start --head 
        --port=6379 
        --num-cpus=4 
        --object-store-memory=1000000000 &&
      sleep infinity
      "
    
    # Health check (ensures container is ready before workers start)
    healthcheck:
      test: ["CMD", "ray", "status"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 30s

  ray-worker:
    # Same image as head
    build:
      context: .
      dockerfile: Dockerfile
    
    # Depends on head being healthy
    depends_on:
      ray-head:
        condition: service_healthy
    
    # Command connects to head node
    command: >
      bash -c "
      ray start 
        --address=ray-head:6379 
        --num-cpus=2 
        --object-store-memory=1000000000 &&
      sleep infinity
      "
    
    # Scale to N worker nodes (we set this to 2)
    deploy:
      replicas: 2
```

**Key Concepts:**
- **Services:** Each service is a separate container
- **build:** Instructions to build the Docker image
- **ports:** Expose container ports to your host machine
- **command:** What to run when the container starts
- **depends_on with health check:** Ensures head is ready before workers start
- **environment:** Pass configuration as environment variables
- **networks:** Automatic network so containers can communicate

---

### Method 1: Run a Single Command

```bash
docker-compose exec ray-head python /app/example-job.py
```

### Method 2: Interactive Shell

```bash
# Open a shell in the head container
docker-compose exec ray-head bash

# Inside the container:
root@ray-head:/# python example-job.py
root@ray-head:/# ray status
root@ray-head:/# exit
```

### Method 3: Submit Jobs Remotely

Edit `example-job.py` to connect to the remote cluster:

```python
import ray

# Connect to the Ray head node
ray.init(address="ray://ray-head:10001")

@ray.remote
def my_task(x):
    return x * x

result = ray.get(my_task.remote(5))
print(result)

ray.shutdown()
```

Then run from your local machine:

```bash
python example-job.py
```

## Monitoring the Cluster

### Ray Dashboard (Web UI)

The Ray Dashboard provides a GUI for cluster monitoring:

```
http://localhost:8265
```

You'll see:
- Cluster configuration (nodes, CPUs, GPUs, memory)
- Running jobs and tasks
- Object store memory usage
- Performance metrics and logs

### Command Line Monitoring

```bash
# Check cluster status
docker-compose exec ray-head ray status

# View Ray logs
docker-compose logs -f ray-head
docker-compose logs -f ray-worker-1

# Monitor resource usage
docker stats

# Enter a container
docker exec -it ray-cluster-mgmt_ray-head_1 bash
```

## Scaling the Cluster

### Add More Worker Nodes at Runtime

```bash
# Scale up to 5 worker nodes
docker-compose up -d --scale ray-worker=5

# Scale down to 1 worker node
docker-compose up -d --scale ray-worker=1

# Verify the change
docker-compose ps
```

## Configuration Options

### Environment Variables (`.env` file)

Create a `.env` file to customize configuration:

```env
# Ray head configuration
RAY_HEAD_PORT=6379
RAY_HEAD_DASHBOARD_PORT=8265

# Ray worker configuration
RAY_WORKER_CPUS=2
RAY_WORKER_MEMORY=2000000000  # 2GB in bytes

# Image configuration
PYTHON_VERSION=3.9
```

### Modifying `docker-compose.yml`

Change resource limits for workers:

```yaml
services:
  ray-worker:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

## Stopping and Cleaning Up

### Pause the Cluster (keeps containers)

```bash
docker-compose pause
docker-compose unpause
```

### Stop the Cluster (removes containers, keeps images)

```bash
docker-compose stop
docker-compose start  # Restart later
```

### Remove Everything

```bash
# Remove containers, networks (keeps images)
docker-compose down

# Remove containers, networks, and volumes
docker-compose down -v

# Remove containers, networks, volumes, and images
docker-compose down -v --rmi all
```

## Troubleshooting

### Port Conflicts

If port 6379 or 8265 is already in use:

```yaml
# Edit docker-compose.yml
services:
  ray-head:
    ports:
      - "6380:6379"     # Map to a different port
      - "8266:8265"
```

### Workers Can't Connect to Head

Ensure all containers are on the same network:

```bash
docker network inspect ray-network

# Check connectivity from a worker
docker-compose exec ray-worker-1 ping ray-head
```

### Out of Disk Space

Clean up unused Docker resources:

```bash
docker system prune -a --volumes
```

### Memory Issues

Increase Docker's memory limit in Docker Desktop settings, or reduce worker count:

```bash
docker-compose up -d --scale ray-worker=2  # Use fewer workers
```

### Logs and Debugging

```bash
# View detailed logs
docker-compose logs --tail=50

# Follow logs in real-time
docker-compose logs -f

# Check a specific service
docker-compose logs ray-head
```

## Custom Workloads

### Add Your Own Script

1. Place your `.py` file in this directory
2. Mount it in `docker-compose.yml`:

```yaml
ray-head:
  volumes:
    - ./my-script.py:/app/my-script.py
```

3. Run it:

```bash
docker-compose exec ray-head python /app/my-script.py
```

### Install Additional Packages

Edit the `Dockerfile` to add packages:

```dockerfile
RUN pip install scikit-learn xgboost
```

Then rebuild:

```bash
docker-compose build
docker-compose up -d
```

## Performance Tips

1. **Use named volumes** for better I/O performance on macOS/Windows.
2. **Increase Docker memory** if workers crash unexpectedly.
3. **Use `--scale`** to adjust worker count based on workload.
4. **Mount code as volumes** to avoid rebuilding images each change.

## Next Steps

1. Modify `example-job.py` for your use case.
2. Add custom packages to the `Dockerfile`.
3. Scale the cluster as needed.
4. Export and deploy the image to a container registry (Docker Hub, ECR, etc.).

## Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Ray in Docker](https://docs.ray.io/en/latest/ray-core/installation.html#docker)
- [Ray Dashboard](https://docs.ray.io/en/latest/ray-core/ray-dashboard.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
