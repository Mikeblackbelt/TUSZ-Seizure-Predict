import os
import numpy as np
from pipeline.checkpoint_io import load_checkpoint, save_checkpoint
from pipeline.eeg_channels import N_TARGET_CHANNELS
from util import handle_logs

logger = handle_logs.get_logger("bipolar_montages", "applog")

# This is based on the adjacent scalp electrode pairs from the standard "double banana" 10-20 format 
BIPOLAR_PAIRS = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'), ('C4', 'T4'),
]

# Strict mapping to the CANONICAL_CHANNELS enforced in raw_eeg_extraction.py
CANONICAL_CHANNELS = [
    'FP1', 'F7', 'T3', 'T5', 'O1', 'FP2', 'F8', 'T4', 'T6', 'O2', 
    'F3', 'C3', 'P3', 'F4', 'C4', 'P4', 'CZ'
]
channel_index_dict = {ch: i for i, ch in enumerate(CANONICAL_CHANNELS)}

def create_bipolar_montages(session_key, checkpoint_dir, bipolar_pairs=BIPOLAR_PAIRS):
    """
    Load a session's raw checkpoint, compute the bipolar montage by subtracting 
    channel pairs, and save it as the processed ('proc') checkpoint.

    Parameters:
        session_key (str): The unique session identifier.
        checkpoint_dir (str): Directory containing checkpoints.
        bipolar_pairs (list): List of tuples defining subtraction pairs.

    Returns:
        np.ndarray of shape (len(BIPOLAR_PAIRS), n_samples), or None on failure.
    """
    montage_rows_tuple_list = [
        (channel_index_dict[ch1], channel_index_dict[ch2]) for ch1, ch2 in bipolar_pairs
    ]

    try:
        # Fetch the raw concatenated checkpoint
        raw_array = load_checkpoint(session_key, checkpoint_dir, stage="raw")
    except FileNotFoundError as e:
        logger.warning(e)
        return None

    if not (raw_array.shape[0] == N_TARGET_CHANNELS and raw_array.shape[1] > 0):
        logger.warning(f"Shape assertion failed for {session_key}: {raw_array.shape}")
        return None

    proc_array = np.zeros((len(montage_rows_tuple_list), raw_array.shape[1]))
    
    for i, (ind_ch1, ind_ch2) in enumerate(montage_rows_tuple_list):
        proc_array[i, :] = raw_array[ind_ch1, :] - raw_array[ind_ch2, :]

    logger.info(f"Calculated bipolar montage for {session_key}: {proc_array.shape}")

    # Save the processed checkpoint
    save_checkpoint(proc_array, session_key, checkpoint_dir, stage="proc")

    return proc_array