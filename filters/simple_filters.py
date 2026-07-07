import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

def bandpass_filter_interval(edf_path, t1, t2, target_pattern_fn, low_cutoff=0.5, high_cutoff=40.0, pad_sec=5, order=4):
    """
    Apply a high-pass and low-pass filter (combined) to only the channels
    matching target_pattern_fn, within the [t1, t2] time interval (seconds).

    Parameters
    ----------
    edf_path : str
        Path to the EDF file.
    t1, t2 : float
        Start and end of the interval to filter, in seconds.
    target_pattern_fn : callable
        Function taking a channel name string, returns True if it should be filtered.
    low_cutoff : float
        High-pass cutoff frequency in Hz (removes drift below this).
    high_cutoff : float
        Low-pass cutoff frequency in Hz (removes content above this).
    pad_sec : float
        Extra context (seconds) taken on each side to absorb filter edge transients.
    order : int
        Butterworth filter order (applied per stage).

    Returns
    -------
    raw : mne.io.Raw
        The Raw object with the specified channels/interval filtered in place.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    fs = raw.info['sfreq']
    data = raw.get_data()  # shape (n_channels, n_samples)
    n_samples = data.shape[1]

    t1_s, t2_s = int(t1 * fs), int(t2 * fs)
    pad_s = int(pad_sec * fs)
    win_start = max(0, t1_s - pad_s)
    win_end = min(n_samples, t2_s + pad_s)

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