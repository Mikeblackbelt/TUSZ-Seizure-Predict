import mne
import pandas as pd
from util import handle_logs
from testing.helpers import *

logger = handle_logs.get_logger("process_signal", "logs/app.log")

def load_edf(path):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    metadata = pd.DataFrame({
        "path": path,
        "channels": [raw.ch_names],
        "sfreq": raw.info["sfreq"],
        "n_samples": raw.n_times,
        "duration_sec": raw.n_times / raw.info["sfreq"]
    })
    return raw, metadata

def split_into_epochs(edf_path, epoch_duration=1):
    """
    Load EDF and split into fixed-length epochs.
    
    Parameters:
        edf_path: Path to EDF file
        epoch_duration: Length of each epoch in seconds 
    
    Returns:
        epochs: MNE Epochs object 
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True)
    
    events = mne.make_fixed_length_events(
        raw, 
        id=1,
        duration=epoch_duration,
        overlap=0  
    )
    
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=1,
        tmin=0,
        tmax=epoch_duration,
        baseline=None,
        preload=True,
        verbose=False
    )
    return epochs

def standardize_channels(raw, metadata):

    channel_map = {}
    for ch in raw.ch_names:
        new_name = ch.replace('EEG ', '')
        # Remove reference suffixes: -LE (linked ears), -REF (average reference), 
        new_name = new_name.replace('-LE', '')
        new_name = new_name.replace('-REF', '')
        channel_map[ch] = new_name
    
    raw.rename_channels(channel_map)

    # These are the 19 standard channels that cover every major brain region
    desired_order = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'T3', 'C3', 'CZ', 'C4', 'T4',
        'T5', 'P3', 'PZ', 'P4', 'T6',
        'O1', 'O2'
    ]
    
    missing = [ch for ch in desired_order if ch not in raw.ch_names]
    
    if missing:
        filename = metadata['path'].iloc[0] if isinstance(metadata['path'], pd.Series) else metadata['path']
        logger.warning(f'{filename} missing {len(missing)} channels: {missing}')
        return None 
    
    raw.reorder_channels(desired_order)
    return raw