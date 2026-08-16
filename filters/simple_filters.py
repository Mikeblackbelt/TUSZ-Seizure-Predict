import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt


def _filter_interval(edf_path, t1, t2, target_pattern_fn, low_cutoff=None, high_cutoff=None, pad_sec=5, order=4):
    """Apply a Butterworth filter to selected channels over a time interval.

    Supports low-pass, high-pass, and band-pass behavior depending on which
    cutoff frequencies are provided.
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
    if low_cutoff is None and high_cutoff is None:
        raise ValueError('At least one cutoff frequency must be provided')
    if low_cutoff is not None and low_cutoff <= 0:
        raise ValueError(f'low_cutoff must be > 0, got {low_cutoff}')
    if high_cutoff is not None and high_cutoff >= nyquist:
        raise ValueError(f'high_cutoff must be < Nyquist ({nyquist} Hz), got {high_cutoff}')
    if low_cutoff is not None and high_cutoff is not None and low_cutoff >= high_cutoff:
        raise ValueError(f'low_cutoff ({low_cutoff}) must be < high_cutoff ({high_cutoff})')

    if low_cutoff is not None and high_cutoff is not None:
        sos = butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=fs, output='sos')
    elif low_cutoff is not None:
        sos = butter(order, low_cutoff, btype='high', fs=fs, output='sos')
    else:
        sos = butter(order, high_cutoff, btype='low', fs=fs, output='sos')

    target_idx = [i for i, ch in enumerate(raw.ch_names) if target_pattern_fn(ch)]

    for ch_idx in target_idx:
        segment = data[ch_idx, win_start:win_end]
        filtered = sosfiltfilt(sos, segment)

        rel_start = t1_s - win_start
        rel_end = t2_s - win_start
        data[ch_idx, t1_s:t2_s] = filtered[rel_start:rel_end]

    raw._data = data
    return raw


def bandpass_filter_raw(raw, low_cutoff=0.5, high_cutoff=40.0, order=4, picks=None):
    """Apply a zero-phase Butterworth bandpass filter to an entire preloaded Raw object."""
    if not raw.preload:
        raise ValueError('raw must be preloaded before filtering')

    fs = raw.info['sfreq']
    nyquist = fs / 2

    if low_cutoff <= 0:
        raise ValueError(f'low_cutoff must be > 0, got {low_cutoff}')
    if high_cutoff >= nyquist:
        raise ValueError(f'high_cutoff must be < Nyquist ({nyquist} Hz), got {high_cutoff}')
    if low_cutoff >= high_cutoff:
        raise ValueError(f'low_cutoff ({low_cutoff}) must be < high_cutoff ({high_cutoff})')

    sos_high = butter(order, low_cutoff, btype='high', fs=fs, output='sos')
    sos_low = butter(order, high_cutoff, btype='low', fs=fs, output='sos')

    data = raw._data
    target_idx = picks if picks is not None else range(data.shape[0])

    for ch_idx in target_idx:
        filtered = sosfiltfilt(sos_high, data[ch_idx])
        filtered = sosfiltfilt(sos_low, filtered)
        data[ch_idx] = filtered

    return raw


def bandpass_filter_interval(edf_path, t1, t2, target_pattern_fn, low_cutoff=0.5, high_cutoff=40.0, pad_sec=5, order=4):
    """Apply a Butterworth bandpass filter to selected channels over a time interval."""
    return _filter_interval(
        edf_path,
        t1,
        t2,
        target_pattern_fn,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        pad_sec=pad_sec,
        order=order,
    )


def lowpass_filter_interval(edf_path, t1, t2, target_pattern_fn, cutoff=0.5, pad_sec=5, order=4):
    """Apply a Butterworth low-pass filter to selected channels over a time interval."""
    return _filter_interval(
        edf_path,
        t1,
        t2,
        target_pattern_fn,
        low_cutoff=None,
        high_cutoff=cutoff,
        pad_sec=pad_sec,
        order=order,
    )


def highpass_filter_interval(edf_path, t1, t2, target_pattern_fn, cutoff=120.0, pad_sec=5, order=4):
    """Apply a Butterworth high-pass filter to selected channels over a time interval."""
    return _filter_interval(
        edf_path,
        t1,
        t2,
        target_pattern_fn,
        low_cutoff=cutoff,
        high_cutoff=None,
        pad_sec=pad_sec,
        order=order,
    )

