import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction


def resample_eeg(data: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """
    Resample EEG data to a target sampling frequency using polyphase
    filtering (anti-aliased, no FFT periodicity assumption).

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples) or (n_samples,)
        Raw EEG data. Time is assumed to be the last axis.
    orig_fs : float
        Original sampling rate (e.g. 256, 250, 512).
    target_fs : float
        Target sampling rate (e.g. 250).

    Returns
    -------
    np.ndarray
        Resampled data, same number of dims as input, with the time axis
        rescaled by target_fs / orig_fs.
    """
    if orig_fs <= 0 or target_fs <= 0:
        raise ValueError(f"Sampling rates must be positive: orig_fs={orig_fs}, target_fs={target_fs}")

    if orig_fs == target_fs:
        return data

    frac = Fraction(target_fs / orig_fs).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    return resample_poly(data, up, down, axis=-1)


def rescale_sample_index(sample_idx: int, orig_fs: float, target_fs: float) -> int:
    """
    Rescale a sample-index-based annotation boundary to match a resampled
    signal. if  annotations are stored as sample indices
    rather than timestamps in seconds.
    """
    return round(sample_idx * (target_fs / orig_fs))