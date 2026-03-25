"""
This module contains functionality to read neural dataset
(calcium florescence or spikes) for 'ucdavis' dataset.

By organization there is a separate .mat file for each session.
Summary of the dataset fields is as follows:
- nneur: number of neurons recorded in the session = 2955
- coh: coherence levels, unique [0, 0.2, 0.4, 0.8] = (1, 185)
- dir: motion direction, unique [0, 30, 60, ...330] = (1, 185)
- regions: 6 regions

- RDKstim_day: raw flourescence data for each trial (n_frames, n_neurons, n_trials) = (38, 2955, 185) 
- RDKspikes_day: inferred spike data for each trial (n_frames, n_neurons, n_trials) = (38, 2955, 185)
- zRDKspikes_day: z-scored inferred spike data for each trial (n_frames, n_neurons, n_trials) = (38, 2955, 185)
- base: baseline activity for each neuron for gray screen (n_frames, n_neurons, n_trials) = (8, 2955, 185)
"""


import os
import h5py
import csv
import scipy.io 
import numpy as np
from pathlib import Path

from .calcium_metadata import CalciumMetaData
from ..base_dataset import BaseDataset, register_dataset
from auditory_cortex import neural_data_dir, NEURAL_DATASETS

# DATASET_NAME = 'calcium'
DATASET_NAME = NEURAL_DATASETS[-1]

@register_dataset(DATASET_NAME)
class CalciumDataset(BaseDataset):
    def __init__(self, sess_id=0, data_dir=None):
        """Initialize the UCDavisDataset with session id and data directory."""
        self.metadata = CalciumMetaData()
        sess_id = int(sess_id)
        self.session_id = sess_id
        self.dataset_name = DATASET_NAME
        if data_dir is None:
            data_dir = Path(neural_data_dir)

        self.data_dir = data_dir
        neural_file = 'Mouse_4512_1L_Day_1_RDK.mat'
        self.file_path = self.data_dir / "data" / neural_file
        self.attr_list = ['RDKspikes_day', 'RDKstim_day', 'base', 'coh', 'dir', 'nneur', 'regions', 'zRDKspikes_day']
        self.data = {}
        with h5py.File(self.file_path, 'r') as file:
            for attr in self.attr_list:
                self.data[attr] = file[attr][()]

        _, self.num_channels, self.n_trials = self.data['RDKspikes_day'].shape      # (n_frames, n_neurons, n_trials)
        
        # self.stim_ids = np.arange(self.n_trials) + 1      # 0 zero trial is reserved for gray screen
        # n_training_trials = int(0.8*self.n_trials)
        # self.training_stim_ids = self.stim_ids[:n_training_trials]
        # self.testing_stim_ids = self.stim_ids[n_training_trials:]

        # separately reading neuron labels..
        self.neurRegionsCSV = self.data_dir / "data" / 'neurRegions.csv'
        area_labels = []
        with open(self.neurRegionsCSV, newline='', encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                area_labels.append(row[0])

        self.neurRegions = np.array(area_labels)   

        self.stim_dir = self.data_dir / 'video_clips' #/ f'repeat_1'
        self.stim_names = sorted(os.listdir(self.stim_dir))

        

    def get_stim_ids(self, *args, **kwargs):
        return self.metadata.get_stim_ids(*args, **kwargs)
    
    def get_training_stim_ids(self, *args, **kwargs):
        return self.metadata.get_training_stim_ids(*args, **kwargs)
    
    def get_testing_stim_ids(self, *args, **kwargs):
        return self.metadata.get_testing_stim_ids(*args, **kwargs)

    def get_stim_audio(self, stim_id, *args, **kwargs):
        # stim_id = stim_id[:-1] + "1"
        arr_uint8 = CalciumDataset.load_4d_array(self.stim_dir / f"{stim_id}.h5")
        arr_float = arr_uint8.astype(np.float32) / 127.5 - 1.0
        return arr_float

    def get_stim_duration(self, *args, **kwargs):
        """Returns duration of the stimulus in seconds"""
        return self.metadata.get_stim_duration(*args, **kwargs)

    def get_sampling_rate(self, *args, **kwargs):
        return self.metadata.get_sampling_rate(*args, **kwargs)
    
    def get_num_bins(self, stim_id, bin_width, mVocs=False):
        """Returns the number of bins for the given stimulus id"""
        duration = self.get_stim_duration(stim_id, mVocs)
        return BaseDataset.calculate_num_bins(duration, bin_width/1000)


    def total_stimuli_duration(self, mVocs=False):
        """Returns the total duration of all the stimuli in the experiment,
        separately for unique and repeated stimuli"""
        stim_ids = self.get_stim_ids(mVocs)
        stim_duration = {}
        for stim_type, stim_ids in stim_ids.items():
            stim_duration[stim_type] = sum([self.get_stim_duration(stim_id, mVocs) for stim_id in stim_ids])
        return stim_duration
    
    def stim_id_to_index(self, stim_id):
        _, co, _, di, r = stim_id.split('_')
        mask1 = self.data['coh'].squeeze() == float(co)
        mask2 = self.data['dir'].squeeze() == float(di)
        mask = mask1 & mask2
        idx = np.where(mask)[0][int(r)-1]       # -1 for 0-based index
        return idx

    def extract_spikes(self, repeated=False, **kwargs):
        """Returns the binned spike counts for all the stimuli
        
        Args:
            repeated: bool = If True, extract spikes for repeated stimuli, otherwise for unique stimuli

        Returns:
            spikes: dict of dict = {stim_id: {channel: spike_counts}}
        """
        stim_group = 'repeated' if repeated else 'unique'
        stim_ids = self.get_stim_ids()[stim_group]
        spikes = {}
        for stim_id in stim_ids:
            # idx = stim_id - 1
            idx = self.stim_id_to_index(stim_id)                      # RDKstim_day, RDKspikes_day, zRDKspikes_day
            stored_array = self.data['RDKstim_day'][8:24,:,idx]       # (n_frames, n_neurons, trials)
            if repeated:
                stored_array = stored_array[None,...]      # (1, n_frames, n_neurons)
            spikes[stim_id] = {ch: stored_array[...,ch] for ch in range(self.num_channels)}

        return spikes
    
    @staticmethod
    def load_4d_array(path):
        with h5py.File(path, "r") as f:
            arr = f["data"][:]  # load as NumPy array
        return arr
    







        
