import argparse
import json
import torch
import subprocess
import os
import socket

import logging
# your own logger is unaffected
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    os.environ["PYTHONWARNINGS"] = "ignore"

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_script", type=str, required=True)
    parser.add_argument("--trf_config", type=str, required=True)
    parser.add_argument("--stim_configs", nargs="+", required=True)
    args = parser.parse_args()

    
    logger.info(f"--- RUNTIME CHECK ---")
    logger.info(f"Actual Hostname: {socket.gethostname()}")
    logger.info(f"Launcher on hostname: {os.environ.get('HOSTNAME')}")
    logger.info(f"GPUs visible here: {torch.cuda.device_count()}")

    
    processes = []
    for gpu_id in range(torch.cuda.device_count()):
        p = subprocess.Popen([
            "python",
            args.worker_script,
            "--trf_config", args.trf_config,
            "--stim_config", args.stim_configs[gpu_id],
            "--gpu_id", str(gpu_id),
        ])
        processes.append(p)

    # wait for all to finish
    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()


