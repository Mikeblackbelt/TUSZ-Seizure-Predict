import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction


def resample_eeg(data: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """
    Resample EEG data to a target sampling frequency using polyphase filtering.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape ``(n_channels, n_samples)`` or ``(n_samples,)``.
        The time axis must be the last axis.
    orig_fs : float
        Original sampling frequency.
    target_fs : float
        Target sampling frequency.
    
    Returns
    -------
    np.ndarray
        Resampled EEG data with the same number of dimensions as ``data``.
    
    Raises
    ------
    ValueError
        If either sampling frequency is less than or equal to zero.
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
    Convert a sample index from the original sampling rate to its corresponding index at the target sampling rate.
    
    Parameters:
        sample_idx (int): Sample index at the original sampling rate.
        orig_fs (float): Original sampling rate in samples per second.
        target_fs (float): Target sampling rate in samples per second.
    
    Returns:
        int: Corresponding sample index at the target sampling rate, rounded to the nearest integer.
    """
    return round(sample_idx * (target_fs / orig_fs))