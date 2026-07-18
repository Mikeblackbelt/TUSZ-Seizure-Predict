import mne
import pandas as pd
from util import handle_logs
from testing.helpers import *

logger = handle_logs.get_logger("process_signal", "logs/app.log")
def load_edf(path):
    """
    Load an EDF file and collect metadata describing its recording.
    
    Parameters:
        path: Path to the EDF file.
    
    Returns:
        A tuple containing the MNE raw recording and a DataFrame with the file path,
        channel names, sampling frequency, sample count, and duration in seconds.
    """
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
    Load an EDF recording and divide it into consecutive fixed-duration epochs.
    
    Parameters:
        edf_path: Path to the EDF file.
        epoch_duration: Duration of each epoch in seconds.
    
    Returns:
        epochs: MNE Epochs object containing the segmented recording.
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
def standardize_channel_name(ch):
    """
    Standardize an EEG channel name for channel matching.
    
    Parameters:
        ch (str): Channel name beginning with ``EEG``.
    
    Returns:
        str or None: The channel name without the ``EEG`` prefix, ``-LE`` or
        ``-REF`` suffixes, and spaces; ``None`` for names without the ``EEG``
        prefix.
    """
    if not ch.startswith('EEG'):
        return None
    # Remove reference suffixes: -LE (linked ears), -REF (average reference)
    new_name = ch.replace('EEG', '')
    new_name = new_name.replace('-LE', '')
    new_name = new_name.replace('-REF', '')
    new_name = new_name.replace(" ", "")
    return new_name

def standardize_channels_names(raw, metadata):
    """
    Standardize EEG channel names and update the associated metadata.
    
    Parameters:
        raw: Raw EEG data whose channel names should be standardized.
        metadata: DataFrame containing the channel names in its first row.
    
    Returns:
        tuple: The updated raw data and metadata.
    """
    channel_map = {}

    eeg_channels = []
    for ch in raw.ch_names:
        new_name = standardize_channel_name(ch)
        if new_name is not None and new_name in standard_channels:
            channel_map[ch] = new_name
            eeg_channels.append(ch)

    # This removes non electrode channels like 'PHOTIC PH'
    raw.pick(eeg_channels) 

    old_metadata_channels = metadata['channels'].iloc[0]
    new_metadata_channels = []
    for ch in old_metadata_channels:
        new_name = standardize_channel_name(ch)
        if new_name is not None and new_name in standard_channels:
            new_metadata_channels.append(new_name)

    metadata['channels'].iloc[0] = new_metadata_channels
    logger.info(f"standardized metadata channels {new_metadata_channels}")

    raw.rename_channels(channel_map)
    
    return raw, metadata

# These are the 19 standard channels that cover every major brain region
standard_channels = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'T3', 'C3', 'CZ', 'C4', 'T4',
        'T5', 'P3', 'PZ', 'P4', 'T6',
        'O1', 'O2'
]

def drop_channels(raw, metadata, desired_order=standard_channels):
    """
    Keep the raw signal channels in the requested set and order.
    
    Parameters:
        raw: Raw signal data containing channel names.
        metadata: DataFrame or mapping containing signal metadata.
        desired_order: Channel names to retain and their required order.
    
    Returns:
        The processed raw signal, or None if a requested channel is missing.
    """

    formatted_raw, _ = standardize_channels_names(raw, metadata)
    raw_missing = [ch for ch in desired_order if ch not in formatted_raw.ch_names]
    raw_extra = [ch for ch in formatted_raw.ch_names if ch not in desired_order]

    if raw_missing:
        filename = metadata['path'].iloc[0] if isinstance(metadata['path'], pd.Series) else metadata['path']
        logger.warning(f'{filename} missing {len(raw_missing)} channels: {raw_missing}')
        return None 
    if raw_extra:
        filename = metadata['path'].iloc[0] if isinstance(metadata['path'], pd.Series) else metadata['path']
        logger.info(f'{filename} has {len(raw_extra)} extra channels: {raw_extra}')
        metadata['channels'].iloc[0] = desired_order
        raw.pick_channels(desired_order)
    
    raw.reorder_channels(desired_order)
    return raw

