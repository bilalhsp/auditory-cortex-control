import os
import gc
import logging
import numpy as np
import torch
import torch.nn as nn

from dorsal_dnns import DorsalNet
from dorsal_dnns.decoder import *

# local imports
from auditory_cortex import utils
from .base_feature_extractor import BaseFeatureExtractor, register_feature_extractor
from auditory_cortex import results_dir, cache_dir

import logging
logger = logging.getLogger(__name__)



def extract_subnet_dict(d):
    out = {}
    for k, v in d.items():
        if k.startswith("fully_connected"):
            continue
        if k.startswith("subnet.") or k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v

    return out

def extract_decoder_dict(d):
    out = {}
    for k, v in d.items():
        if k.startswith("fully_connected"):
            out[k] = v
        else:
            continue
    return out



@register_feature_extractor('dorsalnet')
class DorsalNetExtractor(BaseFeatureExtractor):
    def __init__(self, shuffled=False):
        self.model_name = 'dorsalnet'
        config = utils.load_dnn_config(model_name=self.model_name)
        if config['chkpt_dir'] is not None:
            chkpt_dir = config['chkpt_dir']
        else:
            chkpt_dir = os.path.join(results_dir, 'pretrained_weights')
        saved_checkpoint = config['saved_checkpoint']
        chkpt_path = os.path.join(chkpt_dir, self.model_name, saved_checkpoint)

        checkpoint = torch.load(chkpt_path)
        subnet_dict = extract_subnet_dict(checkpoint)
        model = DorsalNet(False, 32)
        model.load_state_dict(subnet_dict)
        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.inp_pool = torch.nn.AdaptiveAvgPool2d((112, 112))

        super().__init__(model, config, shuffled=shuffled, sampling_rate=config['sampling_rate'])
        noutputs = 5
        nclasses = 72
        nfeats = 32
        start_size = 112
        threed = True
        sz = ((start_size + 1) // 2 + 1) // 2
        self.decoder = Point(noutputs, nclasses, nfeats, threed=threed)

        decoder_check = extract_decoder_dict(checkpoint)
        self.decoder.load_state_dict(decoder_check)
        self.decoder = self.decoder.eval().to(self.device)

    def create_hooks(self):
        """Creates hooks for all the layers in the model."""
        def fn(layer, inp, output):
            output = output.squeeze()
            output = self.avg_pool(output)
            _, T,_ ,_ = output.shape
            feats = output.transpose(0, 1).reshape(T, -1)
            self.features[layer.__name__] = feats.squeeze()
        return fn

    def extract_features(self, stim_audios, sampling_rate=None, stim_durations=None, pad_time=None):
        """
        Returns raw features for all layers of the DNN..!

        Args:
            stim_audios (dict): dictionary of audio inputs for each sentence.
                {stim_id: audio}
            sampling_rate (int): sampling rate of the audio inputs.
            stim_durations (dict): dictionary of sentence durations.
                {stim_id: duration}
            pad_time (float): amount of padding time in seconds.

        Returns:
            dict of dict: read this as features[layer_id][stim_id]
        """
        features = {id:{} for id in self.layer_ids}
        for stim_id, stim in stim_audios.items():

            if pad_time is not None:
                pad_samples = int(pad_time*self.sampling_rate)
                C, _, H, W = stim.shape
                padding = np.zeros((C, pad_samples, H, W))
                stim = np.concatenate([padding, stim], axis=1)

            stim_features = self.get_features(stim)
            for layer_id in self.layer_ids:
                layer_name = self.get_layer_name(layer_id)
                features[layer_id][stim_id] = stim_features[layer_name]
                
            del stim_features
            collected = gc.collect()
        return features


    def fwd_pass(self, stim):
        """
        Forward passes audio input through the model and captures 
        the features in the 'self.features' dict.

        Args:
            aud (ndarray): single 'wav' input of shape (t,) 
        
        Returns:
            input (torch.Tensor): returns the torch Tensor of the input sent passed through the model.
        """
        if not isinstance(stim, torch.Tensor):
            stim = torch.tensor(stim, dtype=torch.float32, device=self.device)
            stim = stim.unsqueeze(dim=0)
        stim = self.inp_pool(stim.squeeze()).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            X = self.model(stim)
            out = self.decoder(X)
        return out
