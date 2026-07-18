from pathlib import Path
import pandas as pd
import numpy as np
import mne
from scipy import signal
from scipy.stats import variation
from scipy.signal import resample_poly
from fractions import Fraction


#uses the methods of Mahdi, M., & Baghdadi, A (2026)

def detect_noise_frequencies(raw, max_freq=60, variance_threshold=0.20, power_percentile=80):
    """
    Identify frequencies with high, consistent power across EEG channels.
    
    Parameters:
        raw: MNE Raw object containing the EEG data.
        max_freq: Upper frequency limit for analysis, in Hz.
        variance_threshold: Maximum cross-channel coefficient of variation for
            considering a frequency uniform.
        power_percentile: Percentile of mean normalized power used to select
            high-power frequencies.
    
    Returns:
        A list of detected noise frequencies in Hz, with nearby frequencies
        grouped by their mean.
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()

    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    if len(eeg_picks) == 0:
        eeg_picks = range(min(20, data.shape[0]))  # fallback: first 20 channels

    # PSD per EEG channel (4-second Welch windows)
    nperseg = min(int(sfreq * 4), data.shape[1])
    all_psds = []
    for i in eeg_picks:
        freqs, psd = signal.welch(data[i], fs=sfreq, nperseg=nperseg)
        all_psds.append(psd)
    all_psds = np.array(all_psds)  # (n_channels, n_freqs)

    # Restrict to frequencies up to max_freq
    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]
    all_psds = all_psds[:, freq_mask]

    # Normalize each channel's PSD
    all_psds_norm = all_psds / (all_psds.sum(axis=1, keepdims=True) + 1e-10)

    # Coefficient of variation per frequency
    # Low CV = power is uniform across channels (noise-like)
    cv_per_freq = variation(all_psds_norm, axis=0)

    mean_power = all_psds_norm.mean(axis=0)
    power_threshold = np.percentile(mean_power, power_percentile)

    noise_freqs = []
    for i, freq in enumerate(freqs):
        # Noise if: high power AND low cross-channel variation
        if mean_power[i] > power_threshold and cv_per_freq[i] < variance_threshold:
            if freq > 4:  # preserve low-frequency delta band
                noise_freqs.append(freq)

    # Merge frequencies within ±1 Hz of each other
    if noise_freqs:
        noise_freqs = np.array(noise_freqs)
        grouped_freqs = []
        i = 0
        while i < len(noise_freqs):
            freq_group = [noise_freqs[i]]
            j = i + 1
            while j < len(noise_freqs) and noise_freqs[j] - noise_freqs[i] < 2:
                freq_group.append(noise_freqs[j])
                j += 1
            grouped_freqs.append(np.mean(freq_group))
            i = j
        noise_freqs = grouped_freqs

    return noise_freqs


def apply_notch_filter(raw, noise_freqs, notch_width=2):
    """
    Apply notch filters at the specified noise frequencies.
    
    Parameters:
        raw: The MNE Raw object to filter.
        noise_freqs: Frequencies, in hertz, at which to apply notch filters.
        notch_width: Width of each notch filter.
    
    Returns:
        A filtered copy of `raw`, or the original object when `noise_freqs` is empty.
    """
    if not noise_freqs:
        return raw

    raw_filtered = raw.copy()
    for freq in noise_freqs:
        raw_filtered.notch_filter(
            freqs=freq,
            picks='all',
            method='spectrum_fit',
            filter_length='auto',
            notch_widths=notch_width,
            verbose=False
        )
    return raw_filtered
