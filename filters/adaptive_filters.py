import numpy as np
from scipy import signal

def compute_channel_psd(eeg_data, fs, nperseg=None):
    """
    This uses welch's method to compute the power spectral density (PSD) for each channel in the EEG data.
    Parameters:
    eeg_data: np.ndarray
        EEG data array of shape (n_channels, n_samples)
    fs: float
        Sampling frequency of the EEG data
    nperseg: int, optional
        Length of each segment for Welch's method. If None, defaults to 4 seconds worth of data or the length of the data, whichever is smaller.
    Returns:
        freqs (n_freqs,), psd (n_channels, n_freqs)
    """
    if nperseg is None:
        nperseg = min(fs * 4, eeg_data.shape[1])  # 4-second windows
    freqs, psd = signal.welch(eeg_data, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd