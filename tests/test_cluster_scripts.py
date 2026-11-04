"""Exercise cluster orchestration without connecting to real computers."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ClusterScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for name in ("start-cluster.sh", "stop-cluster.sh", "status-cluster.sh",
                     "cluster-common.sh", "hosts.txt"):
            shutil.copy(ROOT / name, self.directory / name)
        self.bin = self.directory / "miniconda3/envs/ray-env/bin"
        self.bin.mkdir(parents=True)
        self.log = self.directory / "ssh.log"
        self.env = dict(os.environ, HOME=str(self.directory),
                        PATH=f"{self.bin}:{os.environ['PATH']}",
                        PYTHONPATH=str(self.directory), SSH_LOG=str(self.log),
                        OFFLINE="", CONNECTED="192.168.3.74")
        self.executable("ssh", '''#!/usr/bin/env python3
import os, shlex, subprocess, sys
from pathlib import Path
target, command = sys.argv[-2:]
script = sys.stdin.read() if command.startswith("bash -s") else ""
with open(os.environ["SSH_LOG"], "a") as log:
    log.write(target + " " + command + "\\n")
if target.split("@", 1)[1] in os.environ["OFFLINE"].split():
    print("ssh: connection timed out", file=sys.stderr)
    sys.exit(255)
if 'print(" ".join' in script:
    print(os.environ["CONNECTED"])
    sys.exit(0)
if command.startswith("bash -s"):
    result = subprocess.run(shlex.split(command), input=script, text=True)
else:
    result = subprocess.run(command, shell=True)
sys.exit(result.returncode)
''')
        self.executable("ray", "#!/usr/bin/env bash\nexit 0\n")
        (self.directory / "ray.py").write_text('''
import os
def init(**kwargs): pass
def shutdown(): pass
def nodes():
    addresses = os.environ.get("ADDRESSES", "192.168.3.74 192.168.3.75 192.168.3.76 192.168.3.77").split()
    return [dict(Alive=True, NodeManagerAddress=ip, Resources={"CPU": 32, "GPU": 1})
            for ip in addresses if ip not in os.environ["OFFLINE"].split()]
def cluster_resources():
    return {"CPU": 32 * len(nodes()), "GPU": len(nodes())}
def available_resources(): return cluster_resources()
''')

    def executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def run_script(self, name):
        result = subprocess.run(["bash", str(self.directory / name)],
                                env=self.env, capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=10)
        # Remove failure logs produced by the scripts during simulated outages.
        for line in result.stderr.splitlines():
            if line.strip().startswith("Full details: /tmp/ray-cluster."):
                Path(line.split("Full details: ", 1)[1]).unlink(missing_ok=True)
        return result

    def targets(self):
        return [line.split()[0] for line in self.log.read_text().splitlines()]

    def test_start_all_online(self):
        result = self.run_script("start-cluster.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPU                    0.0 used / 4 total", result.stdout)
        self.assertTrue(result.stdout.rstrip().endswith("Dashboard: http://192.168.3.74:8265"))

    def test_start_uses_detected_resources_on_head_and_workers(self):
        ray_log = self.directory / "ray.log"
        self.env["RAY_LOG"] = str(ray_log)
        self.executable("ray", '''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$RAY_LOG"
[[ "$1" != status ]]
''')
        result = self.run_script("start-cluster.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        starts = [line for line in ray_log.read_text().splitlines()
                  if line.startswith("start ")]
        self.assertEqual(len(starts), 4)
        self.assertTrue(any("--head" in line for line in starts))
        for line in starts:
            for flag in ("--num-cpus", "--num-gpus", "--memory", "--object-store-memory"):
                self.assertNotIn(flag, line)
        self.assertIn("CPU                    0.0 used / 128 total", result.stdout)

    def test_offline_worker_does_not_block_later_workers_or_status(self):
        self.env["OFFLINE"] = "192.168.3.75"
        result = self.run_script("start-cluster.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("pc6@192.168.3.76", self.targets())
        self.assertIn("pc7@192.168.3.77", self.targets())
        self.assertIn("MISSING", result.stdout)
        self.assertTrue(result.stdout.rstrip().endswith("Dashboard: http://192.168.3.74:8265"))

    def test_offline_head_explains_why_workers_cannot_start(self):
        self.env["OFFLINE"] = "192.168.3.74"
        result = self.run_script("start-cluster.sh")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.targets(), ["pc4@192.168.3.74"])
        self.assertIn("workers cannot join", result.stderr)

    def test_stop_continues_after_failures_and_stops_head_last(self):
        self.env["OFFLINE"] = "192.168.3.75 192.168.3.74"
        result = self.run_script("stop-cluster.sh")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.targets(), ["pc5@192.168.3.75", "pc6@192.168.3.76",
                                          "pc7@192.168.3.77", "pc4@192.168.3.74"])

    def test_custom_hosts_drive_start_status_and_stop(self):
        (self.directory / "hosts.txt").write_text(
            "# custom cluster\n\nworker 10.0.0.2 worker\nleader 10.0.0.1 head # head last")
        self.env.update(ADDRESSES="10.0.0.1 10.0.0.2", CONNECTED="10.0.0.1")
        result = self.run_script("start-cluster.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.targets()[0], "leader@10.0.0.1")
        self.assertIn("worker@10.0.0.2", self.targets())
        self.assertIn("Expected: 2", result.stdout)
        self.assertTrue(result.stdout.rstrip().endswith("Dashboard: http://10.0.0.1:8265"))
        self.log.unlink()
        result = self.run_script("stop-cluster.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.targets(), ["worker@10.0.0.2", "leader@10.0.0.1"])

    def test_connected_workers_are_not_started_again(self):
        self.env["CONNECTED"] += " 192.168.3.75"
        result = self.run_script("start-cluster.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("pc5@192.168.3.75", self.targets())
        self.assertIn("already connected", result.stdout)

    def test_invalid_hosts_fail_before_ssh(self):
        for content in ("pc5 192.168.3.75 worker\n",
                        "pc4 192.168.3.74 head\npc5 192.168.3.75 head\n",
                        "pc4 192.168.3.74 head\npc5 192.168.3.74 worker\n",
                        "pc4 192.168.3.74 invalid\n"):
            with self.subTest(content=content):
                (self.directory / "hosts.txt").write_text(content)
                for script in ("start-cluster.sh", "stop-cluster.sh", "status-cluster.sh"):
                    result = self.run_script(script)
                    self.assertEqual(result.returncode, 1)
                    self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
