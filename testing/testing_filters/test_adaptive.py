"""
Unit tests for detect_noise_frequencies() and apply_notch_filter(),
adapted from Mahdi & Baghdadi (2026) - github.com/Mahdyy02/gpt-eeg-transformer"""
import numpy as np
import pytest
import mne
from scipy import signal

from filters.adaptive_filters import detect_noise_frequencies, apply_notch_filter

SFREQ = 250.0
DURATION_SEC = 8.0
N_SAMPLES = int(SFREQ * DURATION_SEC)
N_CHANNELS = 8


def make_raw(data, sfreq=SFREQ, ch_names=None):
    """Wrap a (n_channels, n_samples) array into an MNE RawArray."""
    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(data.shape[0])]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)

def sine(freq, n_samples=N_SAMPLES, sfreq=SFREQ, amplitude=1.0, phase=0.0):
    t = np.arange(n_samples) / sfreq
    return amplitude * np.sin(2 * np.pi * freq * t + phase)

def band_limited_noise(n_samples, sfreq, low, high, rng, amplitude=1.0):
    """Random signal confined to [low, high] Hz, used as 'plausible EEG'."""
    white = rng.standard_normal(n_samples)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, white)
    return amplitude * filtered / (np.std(filtered) + 1e-12)

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)

def test_detects_uniform_60hz_line_noise(rng):
    """A 60 Hz sine injected identically into every channel should be flagged."""
    base = np.stack([
        band_limited_noise(N_SAMPLES, SFREQ, 1, 40, rng, amplitude=1.0)
        for _ in range(N_CHANNELS)
    ])
    line_noise = sine(60, amplitude=5.0)  # same phase/amplitude on every channel
    data = base + line_noise  # broadcasts across channels
    raw = make_raw(data)

    noise_freqs = detect_noise_frequencies(raw, max_freq=80)

    assert len(noise_freqs) > 0
    assert any(abs(f - 60) <= 2 for f in noise_freqs), (
        f"Expected a flag near 60 Hz, got {noise_freqs}"
    )


def test_does_not_flag_channel_localized_signal(rng):
    """A strong 20 Hz component present on only ONE channel (not uniform
    across channels) should NOT be flagged as noise -- it looks like real,
    spatially localized neural activity."""
    base = np.stack([
        band_limited_noise(N_SAMPLES, SFREQ, 1, 40, rng, amplitude=1.0)
        for _ in range(N_CHANNELS)
    ])
    localized = np.zeros_like(base)
    localized[0] += sine(20, amplitude=8.0)  # only channel 0
    data = base + localized
    raw = make_raw(data)

    noise_freqs = detect_noise_frequencies(raw, max_freq=80)

    assert not any(abs(f - 20) <= 2 for f in noise_freqs), (
        f"20 Hz should not be flagged (channel-localized), got {noise_freqs}"
    )


def test_protects_low_frequency_delta_band(rng):
    """Even if a low-frequency (<=4 Hz) component is high-power and uniform
    across channels, it must be excluded per the hardcoded delta-band guard."""
    base = np.stack([
        band_limited_noise(N_SAMPLES, SFREQ, 1, 40, rng, amplitude=1.0)
        for _ in range(N_CHANNELS)
    ])
    delta_noise = sine(2, amplitude=10.0)  # uniform, huge, but <= 4 Hz
    data = base + delta_noise
    raw = make_raw(data)

    noise_freqs = detect_noise_frequencies(raw, max_freq=80)

    assert not any(f <= 4 for f in noise_freqs), (
        f"Frequencies <= 4 Hz must never be flagged, got {noise_freqs}"
    )


def test_no_noise_returns_empty_list(rng):
    """Pure band-limited 'EEG-like' noise with no uniform artifact should
    yield no (or very few incidental) flagged frequencies."""
    data = np.stack([
        band_limited_noise(N_SAMPLES, SFREQ, 1, 40, rng, amplitude=1.0)
        for _ in range(N_CHANNELS)
    ])
    raw = make_raw(data)

    noise_freqs = detect_noise_frequencies(raw, max_freq=80,
                                            variance_threshold=0.20,
                                            power_percentile=80)

    # Independent random noise per channel should rarely satisfy both the
    # high-power AND low-cross-channel-variance condition simultaneously.
    assert len(noise_freqs) <= 1


def test_merges_nearby_flagged_frequencies(rng):
    """Two flagged bins within 2 Hz of each other should collapse into one
    averaged notch target rather than producing two separate entries."""
    base = np.stack([
        band_limited_noise(N_SAMPLES, SFREQ, 1, 40, rng, amplitude=1.0)
        for _ in range(N_CHANNELS)
    ])
    # Two uniform tones 1 Hz apart -- Welch's frequency resolution at this
    # nperseg should place them in adjacent/nearby bins.
    noise = sine(59, amplitude=5.0) + sine(60, amplitude=5.0)
    data = base + noise
    raw = make_raw(data)

    noise_freqs = detect_noise_frequencies(raw, max_freq=80)

    # Should be grouped into a single ~59.5 Hz target, not two entries.
    near_60 = [f for f in noise_freqs if 55 <= f <= 65]
    assert len(near_60) == 1, f"Expected merged single target, got {near_60}"

def test_notch_filter_attenuates_target_frequency(rng):
    base = band_limited_noise(N_SAMPLES, SFREQ, 1, 40, rng, amplitude=1.0)
    data = np.stack([base + sine(60, amplitude=5.0) for _ in range(N_CHANNELS)])
    raw = make_raw(data)

    freqs_before, psd_before = signal.welch(raw.get_data()[0], fs=SFREQ, nperseg=int(SFREQ * 4))
    power_before = psd_before[np.argmin(np.abs(freqs_before - 60))]

    raw_filtered = apply_notch_filter(raw, noise_freqs=[60.0])
    freqs_after, psd_after = signal.welch(raw_filtered.get_data()[0], fs=SFREQ, nperseg=int(SFREQ * 4))
    power_after = psd_after[np.argmin(np.abs(freqs_after - 60))]

    assert power_after < power_before * 0.5, (
        f"Expected substantial attenuation at 60 Hz: before={power_before:.3f}, "
        f"after={power_after:.3f}"
    )


def test_notch_filter_preserves_other_frequencies(rng):
    """Filtering at 60 Hz shouldn't meaningfully touch power at, say, 10 Hz."""
    base = band_limited_noise(N_SAMPLES, SFREQ, 1, 40, rng, amplitude=1.0)
    data = np.stack([base + sine(60, amplitude=5.0) for _ in range(N_CHANNELS)])
    raw = make_raw(data)

    freqs_before, psd_before = signal.welch(raw.get_data()[0], fs=SFREQ, nperseg=int(SFREQ * 4))
    power_before_10hz = psd_before[np.argmin(np.abs(freqs_before - 10))]

    raw_filtered = apply_notch_filter(raw, noise_freqs=[60.0])
    freqs_after, psd_after = signal.welch(raw_filtered.get_data()[0], fs=SFREQ, nperseg=int(SFREQ * 4))
    power_after_10hz = psd_after[np.argmin(np.abs(freqs_after - 10))]

    # Allow some tolerance, but it should not be gutted the way 60 Hz was.
    assert power_after_10hz > power_before_10hz * 0.5


def test_apply_notch_filter_no_op_on_empty_list(rng):
    """If no noise frequencies were detected, the raw object should pass
    through unchanged (same data)."""
    data = np.stack([
        band_limited_noise(N_SAMPLES, SFREQ, 1, 40, rng, amplitude=1.0)
        for _ in range(N_CHANNELS)
    ])
    raw = make_raw(data)

    raw_out = apply_notch_filter(raw, noise_freqs=[])

    np.testing.assert_array_equal(raw_out.get_data(), raw.get_data())


def test_full_pipeline_removes_detected_line_noise(rng):
    base = np.stack([
        band_limited_noise(N_SAMPLES, SFREQ, 1, 40, rng, amplitude=1.0)
        for _ in range(N_CHANNELS)
    ])
    data = base + sine(60, amplitude=5.0)
    raw = make_raw(data)

    detected = detect_noise_frequencies(raw, max_freq=80)
    assert len(detected) > 0

    raw_filtered = apply_notch_filter(raw, detected)

    freqs, psd_before = signal.welch(raw.get_data()[0], fs=SFREQ, nperseg=int(SFREQ * 4))
    _, psd_after = signal.welch(raw_filtered.get_data()[0], fs=SFREQ, nperseg=int(SFREQ * 4))

    idx_60 = np.argmin(np.abs(freqs - 60))
    assert psd_after[idx_60] < psd_before[idx_60] * 0.5