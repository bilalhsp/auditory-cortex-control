"""
Module to build and save stimulus metadata for the UC Davis Act dataset.

This module provides a `StimulusMetadataBuilder` class that constructs a
comprehensive metadata file containing all unique stimulus IDs across
all recording sessions. This is necessary because the original dataset
metadata files may have inconsistencies, and the stimuli presented can
vary slightly from session to session.

The main output is a NumPy `.npz` file (`stim_super_sets.npz`) containing:

    - `timit_ids` : all TIMIT stimulus IDs across sessions
    - `mVocs_ids` : all mVocs stimulus IDs across sessions

Once created, this file can be loaded by a metadata loader class to
quickly access the complete set of stimuli without reading each session
individually. This ensures reproducibility and efficient dataset access
for downstream processing or model training.


Notes:
    - This process needs to run **only once** per dataset installation.
    - The builder reads stimulus IDs from each session’s files,
      merges them into a superset, and saves them as
      a compressed NumPy archive.
    - For string-based stimulus IDs, a fixed-length Unicode dtype is used
      (e.g., `dtype='U32'`) to ensure safe saving and loading without
      pickling.
    - If the metadata file already exists, it does not need to be rebuilt,
      unless sessions have changed or new sessions are added.

Usage Example:

    from auditory_cortex.neural_data.ucdavisAct.metadata_builder import StimulusMetadataBuilder

    builder = StimulusMetadataBuilder()

    # Build and save the stimulus metadata
    builder.build()

    # After this, 'stim_super_sets.npz' will exist on disk and can be used
    # by metadata classes or dataset objects.

"""


from pathlib import Path
import numpy as np

from .ucdavis_dataset import UCDavisActDataset
from .ucdavis_metadata import UCDavisActMetaData

import logging
logger = logging.getLogger(__name__)

class StimulusMetadataBuilder:   
    
    def build(self):
        logger.info(f"Building...")
        curr_dir = Path(__file__).parent
        stim_super_sets_path = curr_dir / 'stim_super_sets.npz'

        stim_sets = self.extract_stim_sets()
        self.save(stim_sets, stim_super_sets_path)

    def extract_stim_sets(self):

        metadata = UCDavisActMetaData()
        timit_ids = set()
        mVocs_ids = set()
        for sess_id in metadata.get_all_available_sessions():
            dataset_obj = UCDavisActDataset(sess_id)

            train_ids = dataset_obj.get_training_stim_ids(mVocs=False)
            test_ids = dataset_obj.get_testing_stim_ids(mVocs=False)
            all_timit_ids = np.concatenate([train_ids, test_ids])
            timit_ids.update(all_timit_ids)

            train_ids = dataset_obj.get_training_stim_ids(mVocs=True)
            test_ids = dataset_obj.get_testing_stim_ids(mVocs=True)
            all_mVocs_ids = np.concatenate([train_ids, test_ids])
            mVocs_ids.update(all_mVocs_ids)

        mVocs_ids = np.array(sorted(mVocs_ids), dtype='U32')
        timit_ids = np.array(sorted(timit_ids), dtype='U32')

        return {
            "mVocs_ids": mVocs_ids,
            "timit_ids": timit_ids
        }

    def save(self, metadata, path):
        logger.info(f"saving metadata to path: {path}")
        np.savez_compressed(path, **metadata)
