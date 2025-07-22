import yaml
import numpy as np
from pathlib import Path
from auditory_cortex import neural_data_dir

class RecordingConfig:
    def __init__(self):
        curr_dir = Path(__file__).parent
        annotations_path = curr_dir / 'sessions_metadata.yml'
        with open(annotations_path, 'r') as f:
            annotations = yaml.safe_load(f)
        self.annotations = annotations

        self.sess_wise_num_repeats = self.annotations['sess_wise_num_repeats']
        self.area_wise_sessions = {
            k: np.array(v) for k,v in self.annotations['area_wise_sessions'].items()
        }
