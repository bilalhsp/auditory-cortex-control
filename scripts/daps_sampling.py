"""
Example Usage:
    python daps_sampling.py --model_name deepspeech2 --layer 2 --lr 0.05
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
import soundfile as sf
from pathlib import Path

# local
from auditory_cortex.diffusion import AudioLDM, Restart, get_sampler
from auditory_cortex.diffusion.utils import *
from auditory_cortex import utils

from auditory_cortex.encoding_models import DeepSpeechsEncoder
from audioldm.utils import seed_everything

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

from auditory_cortex.optimal_stimulus.generator import StimGenerator, save_waveform, Sampler

import argparse
import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()


def peak_normalize(waveform):
    peaks = torch.abs(waveform).max(dim=1, keepdims=True).values
    return waveform / peaks


parser = argparse.ArgumentParser(description='Run reps sampling for a given session and unit')
# parser.add_argument('--session_id', type=int, default=5, help='Session ID to use for sampling')
# parser.add_argument('--unit_id', type=int, default=3002, help='Unit ID to target for sampling')
parser.add_argument('--lr', type=float, default=0.05, help='Learning rate for sampling')
parser.add_argument(
    '-m', '--model_name', dest='model_name', action='store',
    choices=valid_model_names,
    required=True,
    help='model to be used for Regression analysis.'
)
parser.add_argument(
    '-l','--layer', dest='layer_id', type=int, action='store',
    required=True,
    help="Specify the layer ID."
)
parser.add_argument(
    '-s', '--session_id', dest='session_id', type=int, action='store',
    default=5, help="Specify the session ID."
)
# mVocs or timit?
parser.add_argument(
    '-v', '--mVocs', dest='mVocs', action='store_true', default=False,
    help="Whether to use mVocs dataset (default is timit)."
)   
# task: stretch or one-hot
parser.add_argument(
    '-t', '--task', dest='task', type=str, action='store',
    choices=['stretch', 'one-hot'], default='stretch',
    help="Task for optimal stimulus generation: 'stretch' or 'onehot' (default is 'stretch')."
)
# unid id
parser.add_argument(
    '-u', '--unit_id', dest='unit_id', type=int, action='store',
    default=3002, help="Specify the unit ID to target for sampling."
) 

args = parser.parse_args()

# display the arguments passed
for arg in vars(args):
    logging.info(f"{arg:15} : {getattr(args, arg)}")


dataset_name = 'ucdavisAct'
# model_name = 'deepspeech2'
# layer_id = 2

# model_name = 'wav2vec2'
model_name = args.model_name
layer_id = args.layer_id

session_id = args.session_id
mVocs=args.mVocs
task=args.task
unit_id = args.unit_id

base_output_dir = Path("/depot/jgmakin/data/bilal/auditory_opt_stim")

duration=2
n_stimuli=10
bin_width=50
lag=200
generator = StimGenerator(
    dataset_name, model_name, layer_id,
    mVocs=mVocs, bin_width=bin_width, lag=lag
    )



logging.info(f"Loading TRF model for \n\tsession_id: {session_id}, \n\tlayer_id: {layer_id}, \n\tbin_width: {bin_width}, \n\tmVocs: {mVocs}, \n\tlag: {lag}")
trf_model = TRF.load_saved_model(
        model_name, session_id, layer_id, bin_width, mVocs=mVocs,
        tmax=lag, dataset_name=dataset_name
        )

if trf_model is None:
    session_name, corr_dict = generator.fit_encoding_model(session_id)

    trf_model = TRF.load_saved_model(
        model_name, session_id, layer_id, bin_width, mVocs=mVocs,
        tmax=lag, dataset_name=dataset_name
        )
else:
    dataset_obj = create_neural_dataset(dataset_name, session_id)
    generator.data_assembler.read_session_spikes(dataset_obj)
    generator.channel_ids = generator.data_assembler.channel_ids

    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# session_id=5
sampler = Sampler(
    device, model_name=model_name, session_id=session_id, layer_id=layer_id, 
    bin_width=bin_width, mVocs=mVocs, lag=lag, dataset_name=dataset_name, unit_ids=generator.channel_ids
)

test_features, test_spikes = generator.data_assembler.get_testing_data()


predicted_response = trf_model.predict(X=test_features, n_offset=generator.data_assembler.n_offset)


unit_ids = np.array(sampler.channel_ids)
matches = np.where(unit_ids == unit_id)[0]
if len(matches) == 0:
    raise ValueError(
        f"unit_id {unit_id} was not found in the available channel IDs for session {session_id}."
    )
ch_idx = matches[0]
sampler.encoder.set_target(task, ch_idx)
latent_t = compute_latent_t(duration)
z_ch = 8
z_freqs = 16
B = n_stimuli
cond_emb = sampler.audioldm.null_embedding.expand(B, -1, -1)
latent_shape = (B, z_ch, latent_t, z_freqs)
measurement=None

# reps = Restart(self.audioldm, self.encoder, latent=True)
sampler_name = 'daps'
# sampler_config = CONF_DIR / 'reps-config' / 'sampler' / f'{sampler_name}.yaml'
# settings = OmegaConf.load(sampler_config).sampler_config
posterior_sampler = get_sampler(sampler_name, sampler.audioldm, sampler.encoder, latent=True, device=sampler.device)
settings = {
  'sigma_max': 80,
  'sigma_min': 0.1,
  'rho': 15,
  'ode_sigma_min': 0.01,

  'guidance_scale': 1.0,
  'n_anneal_steps': 50,
  'n_ode_steps': 5,

  'lr': args.lr,
  'lr_min_ratio': 0.01,
  'tau': 0.01,
  'n_lang_steps': 50,
}

seed = 42

seed_everything(seed)
extra_settings = {
    'c': cond_emb,
    'shape': latent_shape,
}
settings = {**settings, **extra_settings}
outputs = posterior_sampler.generate_sample(measurement, **settings)[0]

outputs = peak_normalize(outputs)

sub_dir = "lr-{:.1e}".format(args.lr)
stim = 'mVocs' if mVocs else 'timit'

output_dir = (
    base_output_dir
    / sampler_name
    / model_name
    / f"layer-{layer_id}"
    / stim
    / f"session-{session_id}"
    / f"unit-{unit_id}"
    / f"task-{task}"
    / sub_dir
)
output_dir.mkdir(parents=True, exist_ok=True)
logging.info(f"Stimuli saved to: \n {output_dir}")

for i, output in enumerate(outputs):
    output_path = output_dir / f'{task}_stimulus_{i}.wav'
    save_waveform(output_path, output.cpu().numpy(), 16000)