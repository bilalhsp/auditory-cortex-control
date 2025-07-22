"""
This script is used to save the predicted responses of a STRF model
i.e. spectrogram + linear readout. Note that this is not about regression weights, rather
it uses the already saved TRF model (STRF) to compute the predicted responses
of the model on the training and test sets and save them to disk.

Args:
    dataset_name: str ['ucsf', 'ucdavis'], -d
    lag: int, default=200, --lag
    bin_width: int, default=50, -b
    mel_spectrogram: bool, default=False, --mel
    spectrogram_type: str, default='log', --spec_type
    mVocs: bool, default=False, -v

    force_redo: bool, default=False, --force_redo


Usage examples:
    python save_STRF_responses.py -d ucdavis --lag 200 -b 50 --mel --spec_type speech2text
    python save_STRF_responses.py -d ucdavis --lag 200 -b 50 --spec_type cochleogram
"""


import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import gc

# local
from auditory_cortex import saved_corr_dir
import auditory_cortex.utils as utils
import auditory_cortex.io_utils.io as io
from auditory_cortex import valid_model_names

from auditory_cortex.io_utils import ResultsManager
from auditory_cortex.neural_data import create_neural_dataset, create_neural_metadata
from auditory_cortex.dnn_feature_extractor import create_feature_extractor
from auditory_cortex.data_assembler import STRFDataAssembler, DNNDataAssembler
from auditory_cortex.encoding import TRF


def save_predicted_responses(args):

    bin_width = args.bin_width
    lag = args.lag 
    mVocs = args.mVocs
    mel_spectrogram = args.mel_spectrogram
    dataset_name = args.dataset_name
    spectrogram_type = args.spectrogram_type
    force_redo = getattr(args, 'force_redo', False)
    # fixed parameters..
    
    tmin = 0
    num_folds=3

    model_name = f'strf_{spectrogram_type}'
    if mel_spectrogram:
        model_name = 'mel_' + model_name

    metadata = create_neural_metadata(dataset_name)
    neural_dataset = create_neural_dataset(dataset_name)

    data_assembler = STRFDataAssembler(
        neural_dataset, bin_width, 
        mVocs=mVocs,
        mel_spectrogram=mel_spectrogram,
        spectrogram_type=spectrogram_type,
        )
    sessions = metadata.get_all_available_sessions()
    for session in sessions:
        logging.info(f"Working with '{session}'")
        
        if mVocs:
            excluded_sessions = ['190726', '200213']
            if session in excluded_sessions:
                print(f"Excluding session: {session}")
                continue

        predictions = read_training_predictions(
            dataset_name, model_name, session, bin_width,  
            mVocs, lag
            )
        if force_redo or predictions is None:
            logging.info(f"Predicted responses not found for session: {session}...")
            if session != data_assembler.get_session_id():
                # no need to read features again...just reach spikes..
                dataset_obj = create_neural_dataset(dataset_name, session)
                data_assembler.read_session_spikes(dataset_obj)
            
            trf_obj = TRF(model_name, data_assembler)

            corr, opt_lmbda, trf_model = trf_obj.grid_search_CV(
                    lag=lag, tmin=tmin, num_folds=num_folds,
                )
            
            # only returns predicted responses for training set
            predictions = get_predicted_responses(trf_model, data_assembler)

            write_training_predictions(
                predictions, 
                dataset_name, model_name, session, bin_width,  
                mVocs, lag
                )
            
        else:
            logging.info(f"Predicted responses found for session: {session}...")
            

def write_training_predictions(
        predictions, 
        dataset_name, model_name, session, bin_width,  
        mVocs, lag
    ):
    """Writes the training predictions to disk.
    Args:
        predictions: dict, predicted responses of the model on the training set
        dataset_name: str, name of the dataset
        model_name: str, name of the model
        session: str, session ID
        bin_width: int, bin width used for analysis
        mVocs: bool, whether mVocs are used or not
        lag: int, maximum lag used for TRF model

    Returns:
        None
    """
    io.write_result_to_cache(
        predictions, 
        dataset_name=dataset_name, model_name=model_name, session=session,
        bin_width=bin_width, mVocs=mVocs, lag=lag, 
        result_type='training_predictions'
    )

def read_training_predictions(
         dataset_name, model_name, session, bin_width,  
        mVocs, lag
    ):
    """Reads the training predictions from disk.
    Args:
        dataset_name: str, name of the dataset
        model_name: str, name of the model
        session: str, session ID
        bin_width: int, bin width used for analysis
        mVocs: bool, whether mVocs are used or not
        lag: int, maximum lag used for TRF model

    Returns:
        predictions: dict, predicted responses of the model on the training set
    """
    return io.read_result_from_cache(
        dataset_name=dataset_name, model_name=model_name, session=session,
        bin_width=bin_width, mVocs=mVocs, lag=lag, 
        result_type='training_predictions'
    )



def get_predicted_responses(trf_model, data_assembler):
    """Returns the predicted responses of the model on the training and test sets.
    Args:
        trf_model: TRF model object
        data_assembler: Data assembler object containing the features and spikes

    Returns:
        tupe: (training_predictions, test_predictions)
        where,
        training_predictions = {
            'stim_id': {'channel_id': np.ndarray of shape (1, n_samples)} 
        }
    """
    features = data_assembler.data_cache['features']
    train_spikes = data_assembler.data_cache['training_spikes']

    stim_ids = list(train_spikes.keys())
    predicted_spikes = {}
    for stim_id in stim_ids:
        # features[stim_id] = features[stim_id].astype(np.float32)
        stim_features = features[stim_id]   # (n_samples, n_features)
        stim_spikes = train_spikes[stim_id]
        prediction = trf_model.predict([stim_features], data_assembler.n_offset)[0]   # [ndarray of shape: (n_samples, n_channels)]
        predicted_spikes[stim_id] = {channel_id: prediction[:, idx][None, :] for idx, channel_id in enumerate(stim_spikes.keys())}

    return predicted_spikes

# ------------------  get parser ----------------------#

def get_parser():

    parser = argparse.ArgumentParser(
        description='This is to save predicted responses for the model specified',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '-d','--dataset_name', dest='dataset_name', type= str, action='store',
        choices=['ucsf', 'ucdavis'],
        help = "Name of neural data to be used."
    )
    parser.add_argument(
        '--lag', dest='lag', type=int, action='store', 
        default=200,
        help="Specify the maximum lag used for TRF model."
    )
    parser.add_argument(
        '-b','--bin_width', dest='bin_width', type= int, action='store', default=50,
        help="Specify the bin_width to use for analysis."
    )
    parser.add_argument(
        '-v','--mVocs', dest='mVocs', action='store_true', default=False,
        help="Specify if spikes for mVocs are to be used."
    )
    parser.add_argument(
        '--mel', dest='mel_spectrogram', action='store_true', default=False,
        help="Specify if mel_spectrogram to be used as baseline."
    )
    parser.add_argument(
        '--spec_type', dest='spectrogram_type', type=str, action='store',
        help="Specify the type of spectrogram to be used as baseline."
    )
    return parser




# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        logging.info(f"{arg:15} : {getattr(args, arg)}")

    save_predicted_responses(args)
    elapsed_time = time.time() - start_time
    logging.info(f"It took {elapsed_time/60:.1f} min. to run.")