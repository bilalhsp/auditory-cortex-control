import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf

import cupy as cp
import gc

# local
import auditory_cortex
from auditory_cortex.diffusion import AudioLDM, Restart, get_sampler
from auditory_cortex.diffusion.utils import *
from auditory_cortex import utils

from auditory_cortex.encoding_models import DeepSpeechsEncoder

import numpy as np
import matplotlib.pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from auditory_cortex import valid_model_names, NEURAL_DATASETS
import auditory_cortex.io_utils.io as io

from auditory_cortex.neural_data import create_neural_dataset, create_neural_metadata, UCDavisDataset
from auditory_cortex.dnn_feature_extractor import create_feature_extractor
from auditory_cortex.data_assembler import DNNDataAssembler, RandProjAssembler
from auditory_cortex.encoding import TRF


import logging
logger = logging.getLogger(__name__)

CONF_DIR = Path(auditory_cortex.__file__).resolve().parents[1] / 'configs' 

def save_waveform(file_path, waveform, sample_rate):
    if torch.is_tensor(waveform):
        waveform = waveform.cpu().numpy()
    waveform = waveform.clip(-1,1).squeeze()
    sf.write(file_path, waveform, sample_rate, format='WAV', subtype='PCM_16')

class StimGenerator:
    def __init__(
            self, dataset_name, model_name, layer_id, 
            mVocs=False, bin_width=50, lag=200, shuffled=False
        ):

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.layer_id = layer_id
        self.mVocs = mVocs
        self.bin_width = bin_width
        self.lag = lag
        self.shuffled = shuffled

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_models()

    def _load_models(self):
        """Loads everything to vRAM."""
        self.audioldm = AudioLDM(beta_min=0.0015, beta_max=0.0195)
        feature_extractor = create_feature_extractor(self.model_name, shuffled=self.shuffled)
        self.metadata = create_neural_metadata(self.dataset_name)

        # this dataset is placeholder...will be updated to real-time recording..
        neural_dataset = create_neural_dataset(self.dataset_name)
        self.data_assembler = DNNDataAssembler(
                neural_dataset, feature_extractor, self.layer_id, 
                bin_width=self.bin_width, mVocs=self.mVocs,
                )
        logger.info(f"Models and features loaded.")
        
    def fit_encoding_model(self, session_id=0, session_source='existing'):

        full_path=False if session_source =='existing' else True

        dataset_obj = create_neural_dataset(self.dataset_name, session_id, full_path)
        self.data_assembler.read_session_spikes(dataset_obj)
        self.channel_ids = self.data_assembler.channel_ids

        # encoding model is saved to memory for the first time
        # and re-used for multiple runs...
        session_name = dataset_obj.session_name
        # trf_model = TRF.load_saved_model(
        #     self.model_name, session_name, self.layer_id, self.bin_width, mVocs=self.mVocs,
        #     tmax=self.lag, dataset_name=self.dataset_name
        #     )
        # if trf_model is not None:
        #     corr = self.evaluate_model(trf_model)
        # else:
        # lmbdas = np.logspace(-3, 5, 5)
        lmbdas = None
        trf_obj = TRF(self.model_name, self.data_assembler)        
        corr, opt_lmbda, trf_model = trf_obj.grid_search_CV(
                lag=self.lag, num_folds=3, lmbdas=lmbdas,
            )
        self.trf_model = trf_model
        self.encoder = DeepSpeechsEncoder(
            layer_id=self.layer_id, trf_model=trf_model,            
            )
        logger.info(f"Encoding model fit successfully.")
        corr_dict = {ch: corr_value for ch, corr_value in zip(self.data_assembler.channel_ids, corr)}

        return session_name, corr_dict
    
    def evaluate_model(self, trf_model):
        stim_ids, total_duration = self.data_assembler.dataloader.sample_stim_ids_by_duration(
            percent_duration=None, repeated=True, mVocs=self.data_assembler.mVocs
            )
        test_spect_list, all_test_spikes = self.data_assembler.get_testing_data(stim_ids=stim_ids)
        predicted_response = trf_model.predict(X=test_spect_list, n_offset=self.data_assembler.n_offset)
        predicted_response = np.concatenate(predicted_response, axis=0)     # gives (total_time, num_channels)
        all_test_spikes = np.concatenate(all_test_spikes, axis=1)           # gives (n_repeats, total_time, num_channels)

        corr = utils.compute_avg_test_corr(
            all_test_spikes, predicted_response, None)
        return corr
    
    def get_available_sessions(self):
        existing_sessions = {}
        for sess_id in self.metadata.get_all_available_sessions():
            name = self.metadata.full_session_name(sess_id)
            existing_sessions[sess_id] = name.split('_')[1]
        return existing_sessions


    def generate_and_save_stimuli(
            self, output_dir, unit_id=0, task='stretch', 
            duration=2, n_stimuli=1, target_audio=None, seed=42
            ):
        generated_samples = self.generate_stimuli(
            unit_id=unit_id, task=task, duration=duration, n_stimuli=n_stimuli, target_audio=target_audio, seed=seed
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        saved_files = []
        for idx, sample in enumerate(generated_samples):
            filename = f'unit-{unit_id}-{task}-{idx}.wav'
            file_path = output_dir / filename
            save_waveform(file_path, sample, sample_rate=16000)

            saved_files.append({
                "clip_id": idx,
                "filename": filename,
                "path": str(file_path),
            })
        logger.info(f"Samples generated and saved to \n {output_dir}")
        return saved_files

    def generate_stimuli(self, unit_id=0, task='stretch', duration=2, n_stimuli=1, target_audio=None, seed=42):
        if task=='generation':
            assert target_audio is not None, logger.error(f"target audio MUST be provided for generation task!")
            self.encoder.set_target(task)

            padded_x = get_padded_seqs(target_audio)
            measurement = self.encoder(padded_x.to(self.device))
            B = n_stimuli
            z_ch = 8
            latent_t = wav_to_latent_size(padded_x.shape[1])
            z_freqs = 16
            cond_emb = self.audioldm.null_embedding.expand(B, -1, -1)
            latent_shape = (B, z_ch, latent_t, z_freqs)

        else:
            unit_ids = np.array(self.channel_ids)
            matches = np.where(unit_ids == unit_id)[0]
            ch_idx = matches[0] if len(matches) > 0 else 0    # to avoid mismatch for wrong id....better way to handle it?
            self.encoder.set_target(task, ch_idx)
            latent_t = compute_latent_t(duration)
            z_ch = 8
            z_freqs = 16
            B = n_stimuli
            cond_emb = self.audioldm.null_embedding.expand(B, -1, -1)
            latent_shape = (B, z_ch, latent_t, z_freqs)
            measurement=None

        # reps = Restart(self.audioldm, self.encoder, latent=True)
        sampler_name = 'reps'
        sampler_config = CONF_DIR / 'reps-config' / 'sampler' / f'{sampler_name}.yaml'
        sampler = get_sampler(sampler_name, self.audioldm, self.encoder, latent=True)
        settings = OmegaConf.load(sampler_config).sampler_config

        seed_everything(seed)
        extra_settings = {
            'c': cond_emb,
            'shape': latent_shape,
        }
        settings = {**settings, **extra_settings}
        generated_samples = sampler.generate_sample(measurement, **settings)[0]

        return generated_samples

