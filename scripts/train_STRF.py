"""
This script computes and saves regression for STRF 
model.
It uses the TRF class to compute the regression
and saves the results in a CSV file.
It also allows for the option to save the
parameters of the regression model.

Args:
    dataset_name: str ['ucsf', 'ucdavis'], -d
    lag: int, default=200, --lag
    bin_widths: list of int, -b
    identifier: str, default='', -i
    mVocs: bool, default=False, -v
    mel_spectrogram: bool, default=False, --mel
    spectrogram_type: str, default=None, --type
    start: int, default=0, --start, -s
    end: int, default=41, --end, -e
    save_param: bool, default=False, --save_param

Example usage:
    python train_STRF.py -d ucdavis --lag 200 -b 50 -v --spec_type cochleogram -i initial
"""

import logging
from auditory_cortex.utils import set_up_logging
set_up_logging()

import os
import time
import argparse

import numpy as np
import pandas as pd


# local
from auditory_cortex import NEURAL_DATASETS
from auditory_cortex import utils, config, saved_corr_dir
from auditory_cortex.neural_data import create_neural_dataset, create_neural_metadata
from auditory_cortex.data_assembler import STRFDataAssembler
from auditory_cortex.encoding import TRF
from auditory_cortex.io_utils import ResultsManager


# ------------------  Baseline computing function ----------------------#

def compute_and_save_STRF_baseline(args):

    bin_width = args.bin_width
    tmin = 0
    # tmax = args.tmax
    identifier = args.identifier
    # sfreq = 100

    num_freqs = 80
    delay = 0.0
    num_folds=3
    lag = args.lag 
    mVocs = args.mVocs
    mel_spectrogram = args.mel_spectrogram
    dataset_name = args.dataset_name
    spectrogram_type = args.spectrogram_type
    save_param = args.save_param

    results_identifier = ResultsManager.get_run_id(
            dataset_name, bin_width, identifier, mVocs=mVocs, lag=lag,
        )
    if mel_spectrogram:
        if spectrogram_type is None or 'speech2text' in spectrogram_type:
            substr = 'mel_'
        elif 'whisper' in spectrogram_type:
            substr = 'mel_wh_'
        elif 'deepspeech2' in spectrogram_type:
            substr = 'mel_ds_'
        elif 'librosa' in spectrogram_type:
            substr = 'mel_lib_'
        else:
            raise ValueError(f"Unknown spectrogram type: {spectrogram_type}")
        results_identifier = substr + results_identifier
    else:
        if spectrogram_type is None or 'wavlet' in spectrogram_type:
            substr = 'wavlet_'
        elif 'cochleogram' in spectrogram_type:
            substr = 'coch_'
        results_identifier =  substr + results_identifier

    logging.info(f"Results identifier: {results_identifier}")
    csv_file_name = f'STRF_freqs{num_freqs}_{results_identifier}_corr_results.csv'

    # CSV file to save the results at
    file_exists = False
    file_path = os.path.join(saved_corr_dir, csv_file_name)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        file_exists = True

    metadata = create_neural_metadata(dataset_name)
    sessions = metadata.get_all_available_sessions()
    sessions = sessions[args.start_ind:args.end_ind]

    if file_exists:
        sessions_done = data[(data['delay']==delay) & (data['bin_width']==bin_width)]['session'].unique()
        subjects = sessions[np.isin(sessions,sessions_done.astype(int).astype(str), invert=True)]
    else:
        subjects = sessions

    dataset_obj = create_neural_dataset(dataset_name)
    data_assembler = STRFDataAssembler(
        dataset_obj, bin_width, mVocs=mVocs,
        mel_spectrogram=mel_spectrogram,
        spectrogram_type=spectrogram_type,
        )

    for session in subjects:
        if mVocs:
            excluded_sessions = ['190726', '200213']
            if session in excluded_sessions:
                print(f"Excluding session: {session}")
                continue
        logging.info(f"\n Working with '{session}'")

        if session != data_assembler.get_session_id():
            # no need to read features again...just reach spikes..
            dataset_obj = create_neural_dataset(dataset_name, session)
            data_assembler.read_session_spikes(dataset_obj)
            
        model_name = 'strf'
        trf_obj = TRF(model_name, data_assembler)
        
        corr, opt_lmbda, trf_model = trf_obj.grid_search_CV(
                lag=lag, tmin=tmin, num_folds=num_folds,
            )
        
        if save_param:
            trf_obj.save_model_parameters(  # specifying the layer_id = 0 for STRF
                    trf_model, model_name, 0, session, bin_width, shuffled=False,
                LPF=False, mVocs=mVocs, dataset_name=dataset_name, tmax=lag,
                )

        if mVocs:
            mVocs_corr = corr
            timit_corr = np.zeros_like(corr)
        else:
            mVocs_corr = np.zeros_like(corr)
            timit_corr = corr

        channel_ids = data_assembler.channel_ids
        num_channels = len(channel_ids)
        results_dict = {
            'session': num_channels*[session],
            'bin_width': num_channels*[bin_width],
            'delay': num_channels*[delay],
            'channel': channel_ids,
            'test_cc_raw': timit_corr.squeeze(),
            'mVocs_test_cc_raw': mVocs_corr.squeeze(),
            'num_freqs': num_channels*[num_freqs],
            'tmin': num_channels*[tmin],
            'tmax': num_channels*[lag],
            'lmbda': np.log10(opt_lmbda).squeeze(),
            }
        df = utils.write_to_disk(results_dict, file_path)


# ------------------  get parser ----------------------#

def get_parser():
    # create an instance of argument parser
    parser = argparse.ArgumentParser(
        description='This is to compute and save regression results for STRF model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '-d','--dataset_name', dest='dataset_name', type= str, action='store',
        choices=NEURAL_DATASETS,
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
        '-i','--identifier', dest='identifier', type= str, action='store',
        default='',
        help="Specify identifier for saved results."
    )
    parser.add_argument(
        '--mel', dest='mel_spectrogram', action='store_true', default=False,
        help="Specify if mel_spectrogram to be used as baseline."
    )
    parser.add_argument(
        '--spec_type', dest='spectrogram_type', type=str, action='store',
        help="Specify the type of spectrogram to be used as baseline."
    )
    parser.add_argument(
        '-s','--start', dest='start_ind', type=int, action='store', 
        default=0,
        help="Choose sessions starting index to compute results at."
    )
    parser.add_argument(
        '-e','--end', dest='end_ind', type=int, action='store', 
        default=41,
        help="Choose sessions ending index to compute results at."
    )
    parser.add_argument(
        '--save_param', dest='save_param', action='store_true', default=False,
        help="Save the parameters of TRF to disk for future use."
    )
    return parser




# ------------------  main function ----------------------#

if __name__ == '__main__':

    start_time = time.time()
    print("Starting out...")
    parser = get_parser()
    args = parser.parse_args()

    # display the arguments passed
    for arg in vars(args):
        print(f"{arg:15} : {getattr(args, arg)}")

    compute_and_save_STRF_baseline(args)
    elapsed_time = time.time() - start_time
    print(f"It took {elapsed_time/60:.1f} min. to run.")




       
