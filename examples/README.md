## Demonstration Examples of Cluster Scaling. 

### Step 1: The Baseline (1 Core)
You can run this script anywhere, regardless of cluster status, because it completely ignores Ray. 

2. Run the script: 
   ```bash
   python 1_sequential.py
   ```
**What to watch for:** The terminal will hang for about 36 seconds as standard Python forces a single core to process all 500 million operations. 

### Step 2: The Local Hardware limit (32 Cores)
To prove how much power a *single* machine has when unblocked by Python's GIL, we need to run Ray in complete isolation. Because `pc3` is currently acting as your Master Node, we must temporarily stop the cluster daemon so the script doesn't accidentally hijack the full 132-core network.

1. Shut down the cluster daemon on `pc3`:
   ```bash
   ray status
   ray stop
   ```
2. Run the local script:
   ```bash
   python 2_local.py
   ```
**What to watch for:** Ray will spin up a temporary, isolated instance using only `pc3`'s local cores. The execution time should plummet from ~36 seconds down to roughly **10 seconds**. 

### Step 3: The Distributed Cluster (132 Cores)
Now, we reconnect the network and throw all 8 machines at the exact same math problem.

1. Restart your Head Node daemon on `pc3`:
   ```bash
   ray start --head --port=6379 --dashboard-host=0.0.0.0
   ```
   *(Note: You may need to briefly run your `ray start --address='192.168.3.73:6379'` command on your worker nodes so they rejoin the newly started head node).*
2. Run the cluster script:
   ```bash
   python 3_cluster.py
   ```
**What to watch for:** The script will automatically detect the `auto` daemon, lock 1 CPU across all 132 available cores in your network, and return the result in roughly **2.5 seconds**.
