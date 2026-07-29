import mne
import pandas as pd
from util import handle_logs
from testing.helpers import *

logger = handle_logs.get_logger("process_signal", "logs/app.log")

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
