import argparse
import json
import torch

# local 
from auditory_cortex.optimal_stimulus.generator import Sampler

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trf_config", type=str, required=True)
    parser.add_argument("--stim_config", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    args = parser.parse_args()

    # ✅ set device
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    trf_config = load_json(args.trf_config)
    stim_config = load_json(args.stim_config)

    # use device here
    print(f"Using device: {device}")

    sampler = Sampler(device, **trf_config)

    saved_files = sampler.generate_and_save_stimuli(**stim_config)

if __name__ == "__main__":
    main()