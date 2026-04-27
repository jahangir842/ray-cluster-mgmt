# Ray Cluster: Administrator & Operations Guide

As the Cluster Manager for this 8-node Ubuntu GPU architecture, your primary responsibilities are observability, security, and maintaining the physical/virtual health of the environment. 

## 1. Observability: The Ray Dashboard
The Ray Dashboard is your command center. It provides a real-time UI for cluster metrics, actor states, and hardware utilization.

* **Accessing the Dashboard:** By default, the dashboard runs on the Head Node at port `8265`. 
    * *URL:* `http://<HEAD_NODE_IP>:8265` e.g: http://192.168.3.73:8265
* **What to monitor here:**
    * **Machine View:** Check if any of your 8 nodes are reporting high memory pressure or thermal throttling on the GPUs.
    * **Logical View:** See exactly which Python Actors (e.g., vLLM workers) are running and if any have crashed.
    * **Logs View:** Read standard output/error logs directly from the UI without SSH-ing into individual worker nodes.

## 2. Security & Network Architecture
**CRITICAL WARNING:** Ray is designed for trusted networks. It does *not* have built-in authentication or end-to-end encryption for its internal communications. 

If this cluster is hosted on public IPs, you must restrict access using firewalls (e.g., `ufw` on Ubuntu) or VPC Security Groups.

**Ports to Secure (Only allow internal cluster traffic):**
* `6379`: Global Control Store (GCS). The most critical port. Only worker nodes should access this.
* `10001`: Ray Client port. Developers use this to submit scripts remotely.
* `8265`: The Ray Dashboard. Restrict this to your office/VPN IP range.
* `8076` & `8077` (Default): Object Manager ports. Used for nodes transferring shared memory data to each other.

## 3. Log Hunting & Troubleshooting
When a developer complains that their job crashed, or if `ray status` shows a node as "dead," you need to know where to look.

Every time Ray starts, it creates a new session folder in `/tmp/ray/`. 
* **Path:** `/tmp/ray/session_latest/logs/`

**Key Log Files on the Head Node:**
* `gcs_server.out`: If the whole cluster is acting strangely or nodes cannot join, check this file. It logs the core cluster state.
* `raylet.out`: Logs the local node manager. Look here if a specific node is failing to schedule tasks.

**Key Log Files on Worker Nodes:**
* `worker-[id].out` / `worker-[id].err`: If a specific Python job crashed (e.g., a CUDA Out-Of-Memory error in vLLM), the traceback will be here.

## 4. Managing Memory & Object Spilling
With 8 nodes and large LLM workloads, Out-Of-Memory (OOM) errors are your biggest threat. 

If the `object_store_memory` fills up entirely, Ray will attempt to "spill" (save) the excess data to the local hard drive to prevent the node from crashing.
* **Default Spilling Location:** `/tmp/ray/session_latest/sockets/`
* **Manager Action:** Ensure the partition holding `/tmp` on your Ubuntu machines has plenty of fast NVMe storage. If you run out of both RAM *and* disk space, the Raylet daemon will crash, and the node will drop from the cluster.

## 5. Routine Maintenance & Restarts
When you need to patch the underlying Ubuntu OS or update Nvidia/CUDA drivers, you must gracefully drain and restart the nodes.

1. **Stop Worker Nodes First:** SSH into the worker node and run:
   ```bash
   ray stop
   ```
   *Wait for the node to disappear from `ray status` before patching or rebooting.*
2. **Stop the Head Node Last:** Stopping the head node destroys the cluster state. All running jobs will instantly terminate.
   ```bash
   ray stop
   ```
3. **Clearing Stale State:** If the cluster gets into a severely corrupted state (often due to hard power losses), wiping the temporary Ray files on all machines before restarting can fix "ghost" node issues:
   ```bash
   rm -rf /tmp/ray/*
   ```

***

