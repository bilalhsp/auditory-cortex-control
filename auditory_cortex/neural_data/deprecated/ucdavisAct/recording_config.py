import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from auditory_cortex import neural_data_dir

class RecordingConfig:
    def __init__(self):
        curr_dir = Path(__file__).parent
        annotations_path = curr_dir / 'sessions_metadata.yml'
        with open(annotations_path, 'r') as f:
            annotations = yaml.safe_load(f)
        self.info = pd.DataFrame(annotations['sessions_info'])

        # making sure, each session has a unique num_repeats value
        assert self.info.groupby('session')['num_repeats'].nunique().eq(1).all()
        # dictionary for fast lookup
        self.sess_wise_num_repeats = self.info.groupby('session')['num_repeats'].first().to_dict()
        # self.area_wise_sessions = {'all': self.info['id'].to_numpy(dtype=np.int32)}
        self.area_wise_sessions = self.get_area_wise_sessions()

        self.sessions_dict = dict(
            zip(self.info['id'].values,
                self.info['session'].values)
        )

        # self.sess_wise_num_repeats = self.annotations['sess_wise_num_repeats']
        # self.area_wise_sessions = {
        #     k: np.array(v) for k,v in self.annotations['area_wise_sessions'].items()
        # }

    def get_exp_name(self, session_id, mVocs):
        row = self.info[self.info['id'] == session_id].iloc[0]
        if row['interleaved']:                  # BMA# or F1T
            return f"S1T" if row['target_stimuli'] else f"BMA{row['num_repeats']}"
        else:                                   # BMM# or BMT#
            return f"BMM{row['num_repeats']}" if mVocs else f"BMT{row['num_repeats']}"

    def filter_by_num_repeats(self, num_repeats=None):
        if num_repeats is None:
            return self.info['id'].to_list()
        return self.info[self.info['num_repeats'] == num_repeats]['id'].to_list()
    
    def get_area_wise_sessions(self):
        core_areas = ['A1']
        non_primary_areas = ['CL', 'CM']
        all_recorded_areas = self.info['area'].unique().tolist()
        area_session_dict = {}
        area_wise_sessions = {}
        for area in all_recorded_areas:
            area_session_dict[area] = self.info[self.info['area'] == area]['id'].to_list()
            if area in core_areas:
                area_wise_sessions['core'] = area_wise_sessions.get('core', []) + area_session_dict[area]
                area_wise_sessions['all'] = area_wise_sessions.get('all', []) + area_session_dict[area]
            elif area in non_primary_areas:
                area_wise_sessions['non-primary'] = area_wise_sessions.get('non-primary', []) + area_session_dict[area]
                area_wise_sessions['all'] = area_wise_sessions.get('all', []) + area_session_dict[area]
        return area_wise_sessions
