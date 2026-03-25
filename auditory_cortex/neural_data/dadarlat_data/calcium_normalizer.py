from ..normalizer_calculator import NormalizerCalculator
from ..base_dataset import BaseDataset, create_neural_dataset
from ..base_metadata import create_neural_metadata
import numpy as np

import logging
logger = logging.getLogger(__name__)

class CalciumNormalizer(NormalizerCalculator):
    def __init__(self):
        super().__init__('calcium')


    def get_repeated_spikes(self, session, **kwargs):

        dataset = create_neural_dataset(self.dataset_name, session)
        train_spikes = dataset.extract_spikes(repeated=False)
        test_spikes = dataset.extract_spikes(repeated=True)
        all_spikes = {**train_spikes, **test_spikes}

        stim_ids = np.array(list(all_spikes.keys()))
        unique_stim_ids = np.unique([s[:-2] for s in stim_ids])

        repeated_spikes = {}
        all_channels = list(all_spikes[stim_ids[0]].keys())
        for stim_id in unique_stim_ids:
            stim_reps = [s for s in stim_ids if s.startswith(stim_id)]
            stim_spikes = {}
            for ch in all_channels:
                stim_spikes[ch] = np.stack([all_spikes[st][ch].squeeze() for st in stim_reps])
            repeated_spikes[stim_id] = stim_spikes
        return repeated_spikes
    
    def get_testing_stim_duration(self, mVocs=False):
        return 37*2
    
    def get_test_set_ids(self, percent_duration=None, mVocs=False):
        raise NotImplementedError("Calcium dataset does not have a predefined test set.")




