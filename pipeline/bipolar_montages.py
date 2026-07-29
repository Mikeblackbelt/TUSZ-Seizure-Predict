import os
import numpy as np
from pipeline.eeg_channels import CHANNELS_TO_INCLUDE, N_TARGET_CHANNELS
from util import handle_logs

logger = handle_logs.get_logger("bipolar_montages", "applog")

#This is based on the adjacent scalp electrode pairs from the standard "double banana" 10-20 format 
BIPOLAR_PAIRS = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'), ('C4', 'T4'),
]

# Removes the EEG prefix and -REF/-LE suffixes so that REF and LE map to the same index
channel_index_dict = {}
for ch in CHANNELS_TO_INCLUDE:
    clean_name = ch.replace('EEG', '').replace('-REF', '').replace('-LE', '').replace(' ', '')
    if clean_name not in channel_index_dict:
        channel_index_dict[clean_name] = len(channel_index_dict)

def create_bipolar_montages(npy_path, output_path=None,bipolar_pairs=BIPOLAR_PAIRS):
    """
    Read a single session's concatenated .npy file, compute the bipolar
    montage by subtracting channel pairs, and optionally save the result.

    Parameters:
        npy_path: path to a concatenated (N_TARGET_CHANNELS, n_samples) .npy file
        output_path: if given, saves the resulting array here
        bipolar_pairs: a list of tuples with the desired bipolar_pairs, with the 
        adjacent scalp electrode pairs from the standard "double banana" 10-20 format as the default

    Returns:
        np.ndarray of shape (len(BIPOLAR_PAIRS), n_samples), or None on failure
    """

    montage_rows_tuple_list = [
        (channel_index_dict[ch1], channel_index_dict[ch2]) for ch1, ch2 in bipolar_pairs
    ]

    try:
        temp_read_array = np.load(npy_path)
    except FileNotFoundError:
        logger.warning(f"File not found: {npy_path}")
        return None

    if not (temp_read_array.shape[0] == N_TARGET_CHANNELS and temp_read_array.shape[1] > 0):
        logger.warning(f"Shape assertion failed for {npy_path}: {temp_read_array.shape}")
        return None

    temp_write_array = np.zeros((len(montage_rows_tuple_list), temp_read_array.shape[1]))
    for i, (ind_ch1, ind_ch2) in enumerate(montage_rows_tuple_list):
        temp_write_array[i, :] = temp_read_array[ind_ch1, :] - temp_read_array[ind_ch2, :]

    logger.info(f"Calculated bipolar montage for {npy_path}: {temp_write_array.shape}")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, temp_write_array)
        logger.info(f"Saved bipolar montage to {output_path}")

    return temp_write_array

if __name__ == "__main__":
    # Testing code to see if the function works on TUSZ
    # This testing code assumes you've already run the testing code in pipeline.raw_eeg_extraction.py
    from pipeline.bipolar_montages import create_bipolar_montages

    input_dir = "raweeg_output"
    output_dir = "bipolar_montages"

    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    succeeded = 0
    failed = 0
    for filename in npy_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        result = create_bipolar_montages(input_path, output_path=output_path)

    for filename in npy_files:
        raw = np.load(os.path.join(input_dir, filename))
        montage = np.load(os.path.join(output_dir, filename))

        all_correct = True
        for i, (ch1, ch2) in enumerate(BIPOLAR_PAIRS):
            idx1 = channel_index_dict[ch1]
            idx2 = channel_index_dict[ch2]
            expected = raw[idx1, :] - raw[idx2, :]
            if not np.allclose(montage[i, :], expected):
                logger.warning(f"{filename}: MISMATCH at row {i} ({ch1}-{ch2})")
                all_correct = False
            if all_correct:
                logger.info(f"{filename}: Correct")