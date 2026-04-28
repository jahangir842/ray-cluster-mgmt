# Ray Cluster: Administrator & Operations Guide

As the Cluster Manager for this 8-node Ubuntu GPU architecture, your primary responsibilities are observability, security, and maintaining the physical and virtual health of the environment.

---

## 1. Cluster Lifecycle Management

Managing the state of the cluster requires a specific sequence of operations to prevent orphaned processes and corrupted cluster states.

### Starting the Cluster
Always ensure you are in the correct Python environment before initializing the Ray daemon.

```bash
# Ensure the correct conda environment is activated
conda activate ray-env

# Start the head node daemon
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

### Verifying Cluster Health
Once the head node is running, verify that the daemon is active and waiting for worker nodes to join.

```bash
# In the same terminal (or a new SSH session to the head node):
ray status
```

### Draining and Stopping the Cluster
When performing routine maintenance, patching Ubuntu, or rebooting, you must shut down the cluster in the correct order to avoid data corruption.

1.  **Stop Worker Nodes First:** SSH into each worker node and terminate the Ray processes. Wait for the node to disappear from the `ray status` output on the head node before proceeding with patching or rebooting.
    ```bash
    ray stop
    ```
2.  **Stop the Head Node Last:** Stopping the head node destroys the overarching cluster state. All running jobs will instantly terminate.
    ```bash
    ray stop
    ```
3.  **Clear Stale State (Troubleshooting):** If the cluster enters a severely corrupted state (often due to hard power losses or kernel panics), wiping the temporary Ray files on all machines before restarting can fix "ghost" node issues.
    ```bash
    rm -rf /tmp/ray/*
    ```

---

## 2. Observability: The Ray Dashboard

The Ray Dashboard is your operational command center. It provides a real-time UI for cluster metrics, actor states, and hardware utilization.

* **Accessing the Dashboard:** By default, the dashboard runs on the Head Node at port `8265`.
    * **URL:** `http://<HEAD_NODE_IP>:8265` (e.g., `http://192.168.3.73:8265`)

**Key Dashboard Views:**
* **Machine View:** Monitor if any of your 8 nodes are reporting high memory pressure or thermal throttling on the GPUs.
* **Logical View:** Identify exactly which Python Actors (e.g., vLLM workers) are active and quickly spot any that have crashed.
* **Logs View:** Read standard output and error logs directly from the UI, eliminating the need to SSH into individual worker nodes.

---

## 3. Security & Network Architecture

> **CRITICAL WARNING:** Ray is designed for trusted networks. It does *not* have built-in authentication or end-to-end encryption for its internal communications.

If this cluster operates on or near public networks, you must restrict access using firewalls (e.g., `ufw` on Ubuntu) or VPC Security Groups. Allow the following ports **only** for internal cluster traffic or trusted administrative IPs:

| Port | Component | Description |
| :--- | :--- | :--- |
| **`6379`** | Global Control Store (GCS) | The most critical port. Only worker nodes should have access. |
| **`10001`** | Ray Client | Used by developers to submit scripts remotely. |
| **`8265`** | Ray Dashboard | Restrict this strictly to your office network or VPN IP range. |
| **`8076-8077`** | Object Manager | Default ports used by nodes to transfer shared memory data to each other. |

---

## 4. Log Hunting & Troubleshooting

When a developer reports a crashed job, or if `ray status` flags a node as "dead," you must locate the root cause in the node's local files. Every time Ray starts, it creates a new session folder.

* **Standard Log Path:** `/tmp/ray/session_latest/logs/`

### Key Log Files

* **Head Node Logs:**
    * `gcs_server.out`: Check this if the entire cluster is unstable or if nodes cannot join. It logs the core, global cluster state.
    * `raylet.out`: Logs the local node manager. Review this if the head node itself is failing to schedule local tasks.
* **Worker Node Logs:**
    * `worker-[id].out` / `worker-[id].err`: If a specific Python job crashed (e.g., a CUDA Out-Of-Memory error in vLLM), the exact traceback and error messages will be found here.

---

## 5. Managing Memory & Object Spilling

With 8 nodes processing large LLM workloads, Out-Of-Memory (OOM) errors are your primary threat.

If the `object_store_memory` (shared RAM) fills up entirely, Ray will attempt to "spill" (save) the excess data to the local hard drive to prevent the node from crashing entirely.

* **Default Spilling Location:** `/tmp/ray/session_latest/sockets/`
* **Manager Action:** Ensure the partition holding `/tmp` on your Ubuntu machines is backed by plenty of fast NVMe storage. If a node exhausts both its RAM *and* its available disk space, the Raylet daemon will crash, and the node will be dropped from the cluster.
