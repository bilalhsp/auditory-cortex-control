"""

"""

import os
from pathlib import Path
import numpy as np

from ..base_metadata import BaseMetaData, register_metadata
from auditory_cortex import neural_data_dir, NEURAL_DATASETS

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
    )

DATASET_NAME = NEURAL_DATASETS[-1]
DATA_DIR = os.path.join(neural_data_dir, DATASET_NAME)

@register_metadata(DATASET_NAME)
class CalciumMetaData(BaseMetaData):
    def __init__(self):
        self.n_sessions = 1
        self.session_ids = np.array([0])

        self.stim_durations = 2.0 
        self.sampling_rate = 60 # in Hz
        coherences = np.array([0.2, 0.4, 0.8])
        directions = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]).astype(float)
        test_directions = np.array([30, 150]).astype(float)

        repeats = [1, 2, 3, 4, 5]
        coherences = np.array([0.2, 0.4, 0.8])
        directions = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]).astype(float)
        test_directions = np.array([30, 150]).astype(float)

        train_ids = [f"coh_{co:.2f}_dir_{di}_{r}" for r in repeats for co in coherences for di in directions if di not in test_directions]
        train_ids.extend([f"coh_{0.0:.2f}_dir_{0.0}_{r}" for r in repeats])
        test_ids = [f"coh_{co:.2f}_dir_{di}_{r}" for r in repeats for co in coherences for di in test_directions]

        self.training_stim_ids = np.array(train_ids)
        self.testing_stim_ids = np.array(test_ids)


    def get_all_available_sessions(self, num_repeats=None):
        """Returns sessions IDs of all available sessions (with neural data available)
        
        Args: 
            num_repeats (int): Number of repeats of test data, default is 12
        """
        return self.session_ids
    
    def num_repeats_for_sess(self, sess_id, mVocs=False):
        """Returns the number of repeats (of test data) for the given session id
        
        Args:
            sess_id (int): Session ID to get the number of repeats for
            mVocs (bool): Number of repeats for mVocs and TIMIT are the same,
                so this argument is not used, but kept for consistency.

        Returns:
            int: Number of repeats for the given session id
        """
        return 1

    def total_stimuli_duration(self, mVocs=False):
        """Returns the total duration of all the stimuli in the experiment,
        separately for unique and repeated stimuli"""
        stim_ids = self.get_stim_ids(mVocs)
        stim_duration = {}
        for stim_type, stim_ids in stim_ids.items():
            stim_duration[stim_type] = sum([self.get_stim_duration(stim_id, mVocs) for stim_id in stim_ids])
        return stim_duration
    
    def get_stim_ids(self, *args, **kwargs):
        stim_ids = {
            'unique': self.training_stim_ids,
            'repeated': self.testing_stim_ids
        }
        return stim_ids
    
    def get_training_stim_ids(self, *args, **kwargs):
        return self.training_stim_ids
    
    def get_testing_stim_ids(self, *args, **kwargs):
        return self.testing_stim_ids
    

    def get_sampling_rate(self, *args, **kwargs):
        return self.sampling_rate

    def get_stim_audio(self, stim_id, mVocs=False):
        """Reads stim audio for the given stimulus id"""
        pass

    def get_stim_duration(self, *args, **kwargs):
        """Returns duration of the stimulus in seconds"""
        return self.stim_durations
    
