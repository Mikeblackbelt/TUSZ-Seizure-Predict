import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

def bandpass_filter_interval(edf_path, t1, t2, target_pattern_fn, low_cutoff=0.5, high_cutoff=40.0, pad_sec=5, order=4):
    """
    Apply a Butterworth bandpass filter to selected channels over a time interval.
    
    Parameters
    ----------
    edf_path : str
        Path to the EDF file.
    t1, t2 : float
        Start and end of the interval to filter, in seconds.
    target_pattern_fn : callable
        Function that receives a channel name and returns whether that channel
        should be filtered.
    low_cutoff : float, default=0.5
        High-pass cutoff frequency in Hz.
    high_cutoff : float, default=40.0
        Low-pass cutoff frequency in Hz.
    pad_sec : float, default=5
        Duration in seconds to include around the interval during filtering to
        reduce edge transients.
    order : int, default=4
        Butterworth filter order for each filter stage.
    
    Returns
    -------
    mne.io.Raw
        The loaded Raw object with filtered data written to the selected channels
        within the specified interval.
    
    Raises
    ------
    ValueError
        If the cutoff frequencies are invalid.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    fs = raw.info['sfreq']
    data = raw.get_data()  # shape (n_channels, n_samples)
    n_samples = data.shape[1]

    t1_s, t2_s = int(t1 * fs), int(t2 * fs)
    pad_s = int(pad_sec * fs)
    win_start = max(0, t1_s - pad_s)
    win_end = min(n_samples, t2_s + pad_s)

    nyquist = fs / 2
    if low_cutoff <= 0:
        raise ValueError(f"low_cutoff must be > 0, got {low_cutoff}")
    if high_cutoff >= nyquist:
        raise ValueError(f"high_cutoff must be < Nyquist ({nyquist} Hz), got {high_cutoff}")
    if low_cutoff >= high_cutoff:
        raise ValueError(f"low_cutoff ({low_cutoff}) must be < high_cutoff ({high_cutoff})")

    sos_high = butter(order, low_cutoff, btype='high', fs=fs, output='sos')
    sos_low = butter(order, high_cutoff, btype='low', fs=fs, output='sos')

    target_idx = [i for i, ch in enumerate(raw.ch_names) if target_pattern_fn(ch)]

    for ch_idx in target_idx:
        segment = data[ch_idx, win_start:win_end]

        filtered = sosfiltfilt(sos_high, segment)
        filtered = sosfiltfilt(sos_low, filtered)

        rel_start = t1_s - win_start
        rel_end = t2_s - win_start
        data[ch_idx, t1_s:t2_s] = filtered[rel_start:rel_end]

    raw._data = data
    return raw