import torch
import torch.nn.functional as F
from audioldm import utils as aldm_utils 


def seed_everything(seed):
    aldm_utils.seed_everything(seed)

def compute_latent_t(dur):
    sr = 16000
    n_samples = int(dur*sr)
    l_samples = wav_to_latent_size(n_samples)
    desired_l_samples = get_valid_size(l_samples, 8)
    # desired_samples = latent_to_wav_size(desired_l_samples)
    return desired_l_samples

def get_padded_seqs(x):
    """Returns padded inputs to have input lengths as multiple of 
    min_dur
    """
    assert x.ndim == 2, f"Expected 2D inputs, got {x.ndim}-dimensional."
    input_samples = x.shape[1]
    l_samples = wav_to_latent_size(input_samples)
    desired_l_samples = get_valid_size(l_samples, 8)
    desired_samples = latent_to_wav_size(desired_l_samples)
    pad_length = desired_samples - x.shape[1]
    x = F.pad(x, (0, pad_length))
    return x


def get_valid_size(size, min_size):
    """Rounds number of samples to the next valid number of samples
    """
    is_int = (size % min_size) == 0
    if is_int:
        size = size
    else:
        mul = size // min_size + 1
        size = min_size * mul
    return size


def wav_to_latent_size(w_size):
    return int((w_size - 32)/640)

def latent_to_wav_size(l_size):
    return int(l_size*640 + 32)