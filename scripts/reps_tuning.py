"""This script is used to generate samples from 
posterior distribution i.e. p(x|y). 
It uses unconditioned diffusion model as prior p(x) and
measurement model p(y|x) to generate samples.

Args:
    dataset: str ['imagenet'], -d
    problem: str ['super_resolution', 'gaussian_deblur', 'motion_deblur'], -p
    method: str ['dps', 'pc'], -m
    setting_name: int, -s

Example usage:
python generate_posterior_samples.py -d imagenet -p super_resolution -m dps -s 1
"""
# generate_samples.py

# ------------------  set up logging ----------------------
import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

import json
import yaml
import time
import logging
import argparse
import shutil
import hydra
import numpy as np
from omegaconf import OmegaConf

from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


from auditory_cortex import results_dir, cache_dir
from auditory_cortex.diffusion import AudioLDM, Restart, get_sampler
from auditory_cortex.diffusion.utils import *
from auditory_cortex.encoding_models import DeepSpeechsEncoder
from auditory_cortex.diffusion.eval import get_eval_fn, Evaluator

from dotenv import load_dotenv
import os, wandb


@hydra.main(version_base='1.3', config_path="../configs/reps-config", config_name="default")
def main(args):

    output_root_dir = Path(args.output_root_dir)
    if args.wandb:
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        # initialize a run
        wandb.init(
                project=args.project_name,
                name=args.project_name,
                config=OmegaConf.to_container(args, resolve=True, structured_config_mode=False),
                dir=output_root_dir,
            )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.cuda.set_device('cuda:{}'.format(args.gpu))

    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    # dataset and dataloader
    dataset = ESC50Test(**args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    audioldm = AudioLDM(beta_min=0.0015, beta_max=0.0195)
    encoder = DeepSpeechsEncoder(layer_id=2)
    # sampler = Restart(audioldm, encoder, latent=True)
    sampler = get_sampler(args.sampler.name, audioldm, encoder, latent=True)
    
    settings = args.sampler.sampler_config
    # sub_dir = utils.settings_to_dirname(args.sampler.sampler_config)
    # sub_dir = f"NFE-{int(settings['n_restarts']*settings['n_ode_steps'])}"
    # output_dir = Path(output_root_dir, args.data.name, args.task.task_name, args.sampler.name, sub_dir)
    

    # create the sampler
    full_samples = []
    sampling_start_time = time.time()
    for run_id in range(args.num_runs):
        lpips_run, psnr_run, ssim_run = [], [], []
        gen_samples = []
        gt_samples = []
        y_samples = []
        for batch in dataloader:
            wavs, labels = batch
            wavs = wavs.to(device)
            ref_wavs = get_padded_seqs(wavs)
            # generate sample
            measurement = sampler.get_measurement_signal(ref_wavs)
            B = measurement.shape[0]

            z_ch = 8
            latent_t = wav_to_latent_size(ref_wavs.shape[1])
            z_freqs = 16
            kwargs = {
                'c': audioldm.get_text_embedding([""]*B),
                'shape': (B, z_ch, latent_t, z_freqs),
            }
            settings = {**settings, **kwargs}

            generated_samples = sampler.generate_sample(measurement, **settings)[0]

        #     gt_spect = encoder.process_input(ref_wavs)
        #     gen_samples.append(encoder.process_input(generated_samples))
        #     if run_id == 0:
        #         gt_samples.append(gt_spect)
        #         y_samples.append(measurement)

        # full_samples.append(torch.cat(gen_samples, dim=0))
        # if run_id == 0:
        #     samples = torch.cat(gt_samples, dim=0)
        #     y = torch.cat(y_samples, dim=0)

    total_sampling_time = time.time() - sampling_start_time
    # full_samples = torch.stack(full_samples, dim=0)
    # evaluate and log metrics

    # print(f"samples shape: {samples.shape}")
    # print(f"y shape: {y.shape}")
    # print(f"full_samples shape: {full_samples.shape}")

    # results = evaluator.report(samples, y, full_samples) 
    # if args.wandb:
    #     evaluator.log_wandb(results, samples.shape[0])
    # markdown_text = evaluator.display(results) 
    
    # logging.info(f"{markdown_text}")
    logging.info(f"\n Total time for generating samples: {(total_sampling_time)/60:.2f} minutes")  
    logging.info(f"Time per sample: {(total_sampling_time)/args.num_runs/len(dataset):.2f} seconds")  
    

def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)
    return Path(dir)








class ESC50Test(Dataset):
    def __init__(self, sample_rate=16000, duration=3.0, start_idx=0, end_idx=10):
        HF_CACHE_DIR = cache_dir / 'hf_cache'
        hf_dataset = load_dataset(
            "ashraq/esc50",
            split="train",              
            cache_dir=HF_CACHE_DIR,
        )

        # Limit to first 10 samples
        hf_dataset = hf_dataset.select(range(start_idx, end_idx))
        self.ds = hf_dataset
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration)

    def __len__(self):
        return len(self.ds)   # now returns 10

    def __getitem__(self, idx):
        item = self.ds[idx]

        wav = torch.from_numpy(item["audio"]["array"]).float()
        sr = item["audio"]["sampling_rate"]

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav.unsqueeze(0)).squeeze(0)

        if wav.size(0) > self.target_len:
            wav = wav[:self.target_len]
        else:
            wav = F.pad(wav, (0, self.target_len - wav.size(0)))

        label = item["category"]
        return wav, label
    


if __name__ == "__main__":

    START_TIME = time.time()
    main()
    # logging.info(f"Total time taken: {(time.time()-START_TIME)/60} minutes")
    

