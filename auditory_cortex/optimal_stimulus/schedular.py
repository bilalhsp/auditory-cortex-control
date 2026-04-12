import os
import torch
import subprocess
from pathlib import Path
import json

class Schedular:
    def __init__(self, script, trf_config, stim_configs):
        self.script = script
        self.trf_config = trf_config
        self.stim_configs = stim_configs

        self.gpus_per_node = torch.cuda.device_count()

    def get_all_gpu_count(self):
        return self.gpus_per_node*len(self.get_hostlist())

    def get_hostlist(self):
        hosts = subprocess.check_output(["srun", "hostname"], text=True).splitlines()
        hosts = sorted(set(hosts))  
        return hosts
    
    def get_hostname(self):
        return os.environ.get("HOSTNAME")
    
    def launch_jobs_on_current_node(self):
        processes = []
        for gpu_id in range(self.gpus_per_node):
            p = subprocess.Popen([
                "python",
                str(self.script),
                "--trf_config", str(self.trf_config),
                "--stim_config", str(self.stim_configs[gpu_id]),
                "--gpu_id", str(gpu_id),
            ])
            processes.append(p)
        return processes
    
    def launch_on_other_nodes(self):
        processes = []
        for idx, hostname in enumerate(self.get_hostlist()):
            if self.get_hostname() == hostname:
                continue
            p = self.launcher(hostname, global_ids=[(idx*self.gpus_per_node)+local_id for local_id in range(self.gpus_per_node)])
            processes.append(p)
        return processes
    

    def launcher(self, hostname, global_ids):
        ROOT = Path(__file__).resolve().parent
        node_launcher = ROOT / "node_launcher.py"
        return subprocess.Popen([
            "srun",
            "--nodelist", hostname.split('.')[0],
            "-N1",
            "-n1",
            "--exact",                   
            "python",
            node_launcher,
            "--worker_script", str(self.script),
            "--trf_config", str(self.trf_config),
            "--stim_configs", *[str(self.stim_configs[idx]) for idx in global_ids],
        ])
    
    def launch_on_all_gpus(self):

        processes = []
        p = self.launch_jobs_on_current_node()
        processes.extend(p)

        p = self.launch_on_other_nodes()
        processes.extend(p)


        # wait for all to finish
        for p in processes:
            p.wait()