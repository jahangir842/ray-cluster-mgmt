## Demonstration Examples of Cluster Scaling. 

### Example 1: Sequential Python Code (Single Machine, Single CPU Core)

* It will run on sigle CPU thread because standard Python has a built-in safety mechanism called the **Global Interpreter Lock (GIL)**.
* Even if you have 32 cores, only one core is allowed to process math at any given millisecond while the others wait.

2. Run the script: 
   ```bash
   python 1_sequential.py
   ```
**What to watch for:** Note the time at the end of the process. 

### Example 2: Distributed Python Code (Single Machine, Multiple CPU Cores)

* We will use **Ray** without a cluster to overcome Python's GIL.
* We must temporarily stop the cluster daemon so the script can run on a single machine.

1. Shut down the cluster daemon:
   ```bash
   ray status
   ray stop
   ```
2. Run the local script:
   ```bash
   python 2_local.py
   ```
**What to watch for:** Note the time at the end of the process.

### Example 3: Distributed Cluster (Multiple Machines, Multiple CPU Cores)

Now, we reconnect the network and throw all 8 machines at the exact same math problem.

1. Confirm the cluster is running.
   ```bash
   ray status
   ```
   *(Note: You may need to briefly run your `ray start --address='192.168.3.73:6379'` command on your worker nodes so they rejoin the newly started head node).*
2. Run the cluster script:
   ```bash
   python 3_cluster.py
   ```
**What to watch for:** Note the time at the end of the process.
