"""
This script fits STRF predictions and saves
results for layers of DNN models and neural areas.
It differs from run_trf.py in that it fits the predictions
of a strf model rather than the actual spikes.
Once fit, the model is still tested on actual spikes
This is done as a sanity test.

Args:
    dataset_name: str ['ucsf', 'ucdavis'], -d
    model_name: str, -m
    layer_ID: int, -l
    bin_widths: list of int, -b
    lag: int, default=200, --lag
    identifier: str, default='', -i
    mVocs: bool, default=False, -v
    shuffled: bool, default=False, -s
    mel_spectrogram: bool, default=False, --mel
    spectrogram_type: str, default='log', --spec_type


Example usage:
    python fit_strf_predictions.py -d ucsf -m whisper_tiny -b 50 -l 0 -i plos_test -v -s 
"""
# ------------------  set up logging ----------------------
import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

import os
import pandas as pd
import numpy as np
import time
import argparse
import gc

# local
from auditory_cortex import saved_corr_dir
import auditory_cortex.utils as utils
from auditory_cortex.io_utils import ResultsManager, io
from auditory_cortex.io_utils.io import write_lmbdas
from auditory_cortex import valid_model_names

from auditory_cortex.neural_data import create_neural_dataset, create_neural_metadata
from auditory_cortex.dnn_feature_extractor import create_feature_extractor
from auditory_cortex.data_assembler import DNNSTRFPredictionsAssembler
from auditory_cortex.encoding import TRF

def compute_and_save_regression(args):

    # bin_widths = config['bin_widths']
    bin_widths = args.bin_widths
    dataset_name = args.dataset_name
    model_name = args.model_name
    layer_ID = args.layer_ID
    shuffled = args.shuffled
    identifier = args.identifier
    mVocs = args.mVocs
    lag = args.lag
    mel_spectrogram = args.mel_spectrogram
    spectrogram_type = args.spectrogram_type

    
    # fixed parameters..
    
    tmin=0
    delay=0
    N_sents=500
    num_folds=3

    strf_model_name = f'strf_{spectrogram_type}'
    if mel_spectrogram:
        strf_model_name = 'mel_' + strf_model_name

    results_identifier = ResultsManager.get_run_id(
            dataset_name, bin_widths[0], identifier, mVocs=mVocs, shuffled=shuffled, lag=lag,
        )
    csv_file_name = model_name + '_' + results_identifier + '_corr_results.csv'

    # CSV file to save the results at
    file_exists = False
    file_path = os.path.join(saved_corr_dir, csv_file_name)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        file_exists = True

    if shuffled:
        logging.info(f"Running TRF for 'Untrained' networks...")
    else:
        logging.info(f"Running TRF for 'Trained' networks...")

    feature_extractor = create_feature_extractor(model_name, shuffled=shuffled)
    metadata = create_neural_metadata(dataset_name)
    sessions = metadata.get_all_available_sessions()
    sessions = np.sort(sessions)

    for bin_width in bin_widths:
        # Session in data_dir that we do not have results for...
        if file_exists:
            sessions_done = data[
                    (data['delay']==delay) & \
                    (data['bin_width']==bin_width) 
                ]['session'].unique()

            subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
        else:
            subjects = sessions

        if len(subjects) == 0:
            logging.info(f"All sessions already done for bin_width: {bin_width}.")
            continue

        neural_dataset = create_neural_dataset(dataset_name)

        training_predictions = read_training_predictions(
            dataset_name, strf_model_name, neural_dataset.session_id, bin_width, mVocs, lag
            )
        data_assembler = DNNSTRFPredictionsAssembler(
                training_predictions, neural_dataset, feature_extractor, layer_ID, bin_width=bin_width, mVocs=mVocs,
                )
        for session in subjects:
            logging.info(f"Working with '{session}'")
            if mVocs:
                excluded_sessions = ['190726', '200213']
                if session in excluded_sessions:
                    logging.info(f"Excluding session: {session}")
                    continue
            if session != data_assembler.get_session_id():
                # no need to read features again...just reach spikes..
                dataset_obj = create_neural_dataset(dataset_name, session)
                training_predictions = read_training_predictions(
                    dataset_name, strf_model_name, dataset_obj.session_id, bin_width, mVocs, lag
                    )
                data_assembler.read_session_spikes(dataset_obj, training_predictions)
            
            trf_obj = TRF(model_name, data_assembler)
            
            corr, opt_lmbda, trf_model = trf_obj.grid_search_CV(
                    lag=lag, tmin=tmin, num_folds=num_folds,
                )
            
                
            if mVocs:
                mVocs_corr = corr
                timit_corr = np.zeros_like(corr)
            else:
                mVocs_corr = np.zeros_like(corr)
                timit_corr = corr


            channel_ids = data_assembler.channel_ids
            num_channels = len(channel_ids)
            corr_dict = {
                'session': num_channels*[session],
                'layer': num_channels*[layer_ID],
                'channel': channel_ids,
                'bin_width': num_channels*[bin_width],
                'delay': num_channels*[delay],
                'test_cc_raw': timit_corr.squeeze(),
                'normalizer': num_channels*[0.0],  # placeholder for normalizer
                'mVocs_test_cc_raw': mVocs_corr.squeeze(),
                'mVocs_normalizer': num_channels*[0.0],  # placeholder for mVocs normalizer
                'opt_lag': num_channels*[lag],
                'opt_lmbda': np.log10(opt_lmbda).squeeze(),
                'N_sents': num_channels*[N_sents],
                }


            df = utils.write_to_disk(corr_dict, file_path)

            # make sure to delete the objects to free up memory
            del trf_obj
            del trf_model
            gc.collect()


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

# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save regression results for for layers '+
            'of DNN models and neural areas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # add arguments to read from command line
    parser.add_argument(
        '-m', '--model_name', dest='model_name', action='store',
        choices=valid_model_names,
        required=True,
        help='model to be used for Regression analysis.'
    )
    parser.add_argument(
        '-d','--dataset_name', dest='dataset_name', type= str, action='store',
        choices=['ucsf', 'ucdavis'],
        help = "Name of neural data to be used."
    )
    parser.add_argument(
        '-b','--bin_widths', dest='bin_widths', nargs='+', type= int, action='store',
        required=True,
        help="Specify list of bin_widths."
    )
    parser.add_argument(
        '--lag', dest='lag', type=int, action='store', 
        default=200,
        help="Specify the maximum lag used for STRF."
    )
    parser.add_argument(
        '-l','--layer', dest='layer_ID', type=int, action='store',
        required=True,
        help="Specify the layer ID."
    )
    parser.add_argument(
        '-i','--identifier', dest='identifier', type= str, action='store',
        default='',
        help="Specify identifier for saved results."
    )
    parser.add_argument(
        '-s','--shuffle', dest='shuffled', action='store_true', default=False,
        help="Specify if shuffled (untrained) network to be used."
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

    compute_and_save_regression(args)
    elapsed_time = time.time() - start_time
    logging.info(f"It took {elapsed_time/60:.1f} min. to run.")