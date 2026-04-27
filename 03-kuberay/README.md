# KubeRay: Ray on Kubernetes

This guide shows how to deploy a production-grade Ray cluster on Kubernetes using KubeRay—the cloud-native approach for enterprise environments.

## Overview

KubeRay is the official Kubernetes operator for Ray that provides:
- **Cloud-Native Deployment:** Deploy Ray clusters as Kubernetes Custom Resources
- **Auto-scaling:** Dynamically add/remove worker nodes based on workload demand
- **Multi-tenancy:** Run multiple Ray clusters on the same Kubernetes infrastructure
- **Observability:** Integration with Prometheus, Grafana, and cloud monitoring tools
- **High Availability:** Automatic node recovery and rolling updates

**When to use KubeRay:**
- Production environments
- Cloud platforms (AWS, GCP, Azure)
- Multi-tenant shared infrastructure
- Need for auto-scaling and cost optimization
- Enterprise monitoring and compliance requirements

## Prerequisites

1. **Kubernetes Cluster (1.24+)**
   - Minikube, kind, or cloud K8s (EKS, GKE, AKS)
   - At least 4 vCPU and 8GB RAM
   - `kubectl` configured and authenticated

2. **Install KubeRay Operator** (one-time setup):

```bash
# Option A: Using Helm (recommended)
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --namespace kuberay-system --create-namespace

# Option B: Using kubectl directly
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/master/helm/kuberay-operator/crds.yaml
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/master/helm/kuberay-operator/templates/operator.yaml
```

3. **Verify Operator Installation:**

```bash
kubectl get pods -n kuberay-system

# Output should show:
# NAME                                      READY   STATUS    RESTARTS   AGE
# kuberay-operator-xxxxx                    1/1     Running   0          5m
```

## Repository Structure

```
03-kuberay/
├── README.md                    # This file
├── ray-cluster.yaml             # Ray cluster definition (CRD)
├── ray-autoscaler.yaml          # Auto-scaler configuration
├── sample-job.yaml              # Example Kubernetes Job
└── sample-raycluster-job.yaml   # Job that runs inside Ray
```

## Quick Start (If Kubernetes is Already Running)

### Step 1: Install KubeRay Operator

The KubeRay operator manages Ray clusters on Kubernetes:

```bash
# Install using Helm (recommended)
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install the operator in a dedicated namespace
helm install kuberay-operator kuberay/kuberay-operator \
  --namespace kuberay-system \
  --create-namespace
```

Verify installation:

```bash
# Check operator is running
kubectl get pods -n kuberay-system
```

Expected output:
```
NAME                                      READY   STATUS    RESTARTS   AGE
kuberay-operator-xxxxx                    1/1     Running   0          2m
```

### Step 2: Deploy a Ray Cluster

First, review [ray-cluster.yaml](ray-cluster.yaml) to understand the configuration:
- **Head node:** 1 replica with 2 CPUs and 4GB memory
- **Worker group:** 2-10 replicas with 2 CPUs and 4GB memory each
- **Auto-scaling:** Enabled to scale workers based on demand

Deploy the cluster:

```bash
# Create the RayCluster resource
kubectl apply -f ray-cluster.yaml
```

This command:
- Parses the YAML file
- Creates a RayCluster resource in Kubernetes
- Operator detects it and starts creating pods
- Takes 30-60 seconds for all pods to be ready

### Step 3: Monitor Cluster Startup

```bash
# Watch cluster status in real-time
kubectl get raycluster -w
```

Wait for `READY` column to show `True`:
```
NAME              READY   AGE
my-ray-cluster    True    45s
```

### Step 4: Access the Ray Dashboard

```bash
# Port-forward the dashboard to your local machine
kubectl port-forward svc/my-ray-cluster-head-svc 8265:8265 &

# Open in browser: http://localhost:8265
```

### Step 5: Submit a Job

```bash
# Method 1: Run a Python script in the head pod
kubectl exec -it $(kubectl get pods -l ray.io/node-type=head \
  -l ray.io/cluster=my-ray-cluster -o jsonpath='{.items[0].metadata.name}') \
  -- python -c "
import ray
ray.init(address='ray://localhost:10001')

@ray.remote
def task(x):
    return x * 2

result = ray.get(task.remote(21))
print(f'Result: {result}')
ray.shutdown()
"
```

Or, submit a Kubernetes Job:

```bash
# Create and submit a job
kubectl apply -f sample-job.yaml

# Monitor job execution
kubectl logs -f job/my-ray-job
```

### Step 6: Clean Up

```bash
# Delete the Ray cluster
kubectl delete raycluster my-ray-cluster

# Wait for pods to terminate
kubectl get pods -w

# (Optional) Uninstall the operator
helm uninstall kuberay-operator -n kuberay-system
```

---

## Understanding the RayCluster CRD (Custom Resource Definition)

The `ray-cluster.yaml` file defines a Ray cluster using Kubernetes Custom Resources. Let's understand each section:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: my-ray-cluster
  namespace: default
spec:
  rayVersion: '2.0.1'    # Ray version to use
  
  # Head node configuration (the coordinator)
  head:
    replicas: 1          # Only 1 head node (required)
    rayStartParams:
      port: '6379'       # Ray client port
      num-cpus: '2'      # CPUs per head
      object-store-memory: '1000000000'  # 1GB memory for data
    
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.0.1
          resources:
            requests:
              cpu: "1"         # Kubernetes will reserve 1 CPU
              memory: "2Gi"    # Reserve 2GB
            limits:
              cpu: "2"         # Allow up to 2 CPUs burst
              memory: "4Gi"    # Allow up to 4GB burst
  
  # Worker node pools (where your tasks run)
  workerGroupSpecs:
  - groupName: "worker-group"
    replicas: 2            # Start with 2 workers
    minReplicas: 2         # Minimum for auto-scaling
    maxReplicas: 10        # Maximum for auto-scaling
    rayStartParams:
      num-cpus: '2'
      object-store-memory: '1000000000'
    
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.0.1
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
  
  # Auto-scaling configuration
  autoscalerOptions:
    upscalingMode: "default"  # Aggressive adds more workers faster
    idleTimeoutSeconds: 60    # Scale down after 60s of no tasks
    resources:
      limits:
        cpu: "500m"           # Operator can use 500m CPUs
```

**Key Concepts:**

- **Replicas:** Number of pods to create
- **minReplicas/maxReplicas:** Auto-scaler bounds
- **requests:** Resources guaranteed to the pod
- **limits:** Maximum resources a pod can use
- **rayStartParams:** Ray configuration options
- **Auto-scaling:** Automatically add/remove workers based on load

---

### Head Node Configuration

```yaml
spec:
  head:
    replicas: 1
    rayStartParams:
      port: '6379'
      num-cpus: '2'
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:latest
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
```

### Worker Node Configuration

```yaml
spec:
  workerGroupSpecs:
  - groupName: "worker-group"
    replicas: 2
    minReplicas: 2
    maxReplicas: 10
    rayStartParams:
      num-cpus: '2'
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:latest
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
```

### Auto-Scaling Configuration

Enable Kubernetes-based auto-scaling:

```yaml
spec:
  autoscalerOptions:
    upscalingMode: "default"  # or "aggressive"
    idleTimeoutSeconds: 60    # scale down after idle
    resources:
      limits:
        cpu: "500m"
```

## Monitoring and Observability

### Ray Dashboard

```bash
# Port-forward and access the dashboard
kubectl port-forward svc/my-ray-cluster-head-svc 8265:8265
# Open: http://localhost:8265
```

### Cluster Status

```bash
# Get cluster status
kubectl get raycluster

# Describe cluster details
kubectl describe raycluster my-ray-cluster

# Get head pod logs
kubectl logs -f deployment/my-ray-cluster-head-deployment

# Get worker pod logs
kubectl logs -f <worker-pod-name>
```

### Event Monitoring

```bash
# Watch cluster events
kubectl get events --sort-by='.lastTimestamp'

# Watch pod creation/deletion
kubectl get pods -w -l ray.io/cluster=my-ray-cluster
```

## Submitting Jobs

### Method 1: Kubernetes Job (Recommended)

Create a `job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: my-ray-job
spec:
  template:
    spec:
      serviceAccountName: default
      containers:
      - name: ray-job
        image: rayproject/ray:latest
        command: ["python", "-c", "import ray; ray.init(address='ray://my-ray-cluster-head:10001'); ..."]
      restartPolicy: Never
  backoffLimit: 3
```

Submit:

```bash
kubectl apply -f job.yaml
kubectl logs -f job/my-ray-job
```

### Method 2: Ray Job API (Experimental)

```bash
# Direct submission via Ray CLI
ray job submit --address ray://my-ray-cluster-head:10001 --runtime-env-json='{"working_dir": "."}' -- python example-job.py
```

### Method 3: Interactive Shell

```bash
# Access head pod shell
kubectl exec -it <head-pod-name> -- bash

# Run Python directly
ray status
python example-job.py
```

## Scaling the Cluster

### Manual Horizontal Scaling

```bash
# Edit the cluster
kubectl edit raycluster my-ray-cluster

# Change worker replicas:
# spec:
#   workerGroupSpecs:
#   - replicas: 5  # Change this number
```

Or, patch directly:

```bash
kubectl patch raycluster my-ray-cluster --type='json' \
  -p='[{"op": "replace", "path": "/spec/workerGroupSpecs/0/replicas", "value":5}]'
```

### Auto-Scaling

Enable auto-scaler in `ray-cluster.yaml`:

```yaml
spec:
  autoscalerOptions:
    upscalingMode: "default"
    idleTimeoutSeconds: 60
```

Ray will automatically scale workers based on task demand.

## Troubleshooting

### Cluster Won't Start

```bash
# Check operator logs
kubectl logs -f deployment/kuberay-operator -n kuberay-system

# Check RayCluster events
kubectl describe raycluster my-ray-cluster

# Check pod events
kubectl describe pod <pod-name>
```

### Nodes Can't Connect

```bash
# Verify network connectivity
kubectl exec -it <head-pod-name> -- ping <worker-pod-name>

# Check DNS
kubectl exec -it <head-pod-name> -- nslookup my-ray-cluster-worker-0
```

### Pod Stuck in Pending

```bash
# Check node resources
kubectl top nodes
kubectl describe nodes

# Check PVC status (if using persistent volumes)
kubectl get pvc
```

### Memory/CPU Limits Exceeded

```bash
# Check resource usage
kubectl top pods -l ray.io/cluster=my-ray-cluster

# Increase requests/limits in ray-cluster.yaml
# and reapply the manifest
```

## Advanced Features

### Custom Ray Images

```yaml
spec:
  head:
    template:
      spec:
        containers:
        - image: myregistry.azurecr.io/ray:latest
```

### Persistent Volumes

```yaml
spec:
  head:
    template:
      spec:
        volumes:
        - name: data
          persistentVolumeClaim:
            claimName: ray-data
```

### Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ray-quota
  namespace: default
spec:
  hard:
    requests.cpu: "20"
    requests.memory: "40Gi"
    limits.cpu: "40"
    limits.memory: "80Gi"
```

## Cleanup

### Delete a Single Ray Cluster

```bash
kubectl delete raycluster my-ray-cluster
```

### Uninstall KubeRay Operator

```bash
# Using Helm
helm uninstall kuberay-operator -n kuberay-system

# Using kubectl
kubectl delete -f https://raw.githubusercontent.com/ray-project/kuberay/master/helm/kuberay-operator/crds.yaml
```

## Best Practices

1. **Use Clear Naming:** Name clusters descriptively (`ml-training-prod`, `inference-staging`).
2. **Set Resource Requests/Limits:** Ensure fair allocation and prevent starvation.
3. **Use Namespaces:** Isolate teams/projects with Kubernetes namespaces.
4. **Monitor with Prometheus:** Export metrics for alerting and dashboards.
5. **Use ConfigMaps for Configuration:** Don't hardcode settings in CRDs.
6. **Implement RBAC:** Restrict cluster access with Kubernetes RBAC.
7. **Plan for Cost:** Use auto-scaling and spot instances to optimize costs.
8. **Backup Data:** Use persistent volumes for important data.

## Next Steps

1. Customize `ray-cluster.yaml` for your use case (add GPUs, increase replicas, etc.).
2. Deploy your Ray job using sample-job.yaml as a template.
3. Set up monitoring with Prometheus and Grafana.
4. Integrate with your CI/CD pipeline.
5. Explore Ray's advanced features (Ray Serve, Ray Tune, Ray Train).

## Resources

- [KubeRay Official Documentation](https://docs.ray.io/en/latest/cluster/kubernetes/)
- [KubeRay GitHub](https://github.com/ray-project/kuberay)
- [Ray Kubernetes Deployment Examples](https://github.com/ray-project/kuberay/tree/master/examples)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [KubeRay RayCluster CRD Reference](https://github.com/ray-project/kuberay/blob/master/proto/kuberay.proto)
