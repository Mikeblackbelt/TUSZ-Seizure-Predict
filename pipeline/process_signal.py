import mne
import pandas as pd

def load_edf_metadata(path):
    raw = mne.io.read_raw_edf(path, preload=False)
    metadata = pd.DataFrame({
        "channels": raw.ch_names,
        "sfreq": raw.info["sfreq"],
        "n_samples": raw.n_times,
        "duration_sec": raw.n_times / raw.info["sfreq"]
    })
    return metadata

def split_into_epochs(edf_path, epoch_duration=30):
    """
    Load EDF and split into fixed-length epochs.
    
    Parameters:
        edf_path: Path to EDF file
        epoch_duration: Length of each epoch in seconds (default: 30)
    
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

