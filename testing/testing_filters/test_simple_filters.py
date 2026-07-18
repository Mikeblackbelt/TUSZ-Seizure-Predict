"""Unit tests for filters.simple_filters.bandpass_filter_interval."""
import os
import tempfile

import mne
import numpy as np
import pytest
from scipy import signal

from filters.simple_filters import bandpass_filter_interval
from util import handle_logs

logger = handle_logs.get_logger("test_simple_filters", "logs/test.log")

SFREQ = 256.0
DURATION_SEC = 20.0
N_SAMPLES = int(SFREQ * DURATION_SEC)


@pytest.fixture
def edf_path():
    """Write a synthetic 2-channel EDF (one target, one non-target) to a
    temporary file and yield its path."""
    t = np.arange(N_SAMPLES) / SFREQ
    target = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 60 * t)
    other = np.sin(2 * np.pi * 5 * t)
    data = np.stack([target, other])

    info = mne.create_info(["TARGET", "OTHER"], SFREQ, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(data, info, verbose=False)

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        raw.export(tmp.name, overwrite=True)
        tmp_path = tmp.name

    yield tmp_path

    os.unlink(tmp_path)


def only_target(ch_name):
    return ch_name == "TARGET"


def none_match(ch_name):
    return False


def test_returns_raw_object(edf_path):
    logger.info("test_returns_raw_object: start")
    result = bandpass_filter_interval(edf_path, t1=5, t2=15, target_pattern_fn=only_target)
    assert isinstance(result, mne.io.BaseRaw)
    logger.info("test_returns_raw_object: passed")


def test_samples_outside_interval_are_unchanged(edf_path):
    logger.info("test_samples_outside_interval_are_unchanged: start")
    before = mne.io.read_raw_edf(edf_path, preload=True, verbose=False).get_data()
    after = bandpass_filter_interval(edf_path, t1=5, t2=15, target_pattern_fn=only_target).get_data()

    fs = SFREQ
    t1_s, t2_s = int(5 * fs), int(15 * fs)
    np.testing.assert_array_equal(after[:, :t1_s], before[:, :t1_s])
    np.testing.assert_array_equal(after[:, t2_s:], before[:, t2_s:])
    logger.info("test_samples_outside_interval_are_unchanged: passed")


def test_non_target_channel_is_never_modified(edf_path):
    logger.info("test_non_target_channel_is_never_modified: start")
    before = mne.io.read_raw_edf(edf_path, preload=True, verbose=False).get_data()
    after = bandpass_filter_interval(edf_path, t1=5, t2=15, target_pattern_fn=only_target).get_data()

    other_idx = 1  # "OTHER" channel
    np.testing.assert_allclose(after[other_idx], before[other_idx], atol=1e-6)
    logger.info("test_non_target_channel_is_never_modified: passed")


def test_no_target_pattern_match_leaves_data_unchanged(edf_path):
    logger.info("test_no_target_pattern_match_leaves_data_unchanged: start")
    before = mne.io.read_raw_edf(edf_path, preload=True, verbose=False).get_data()
    after = bandpass_filter_interval(edf_path, t1=5, t2=15, target_pattern_fn=none_match).get_data()

    np.testing.assert_allclose(after, before, atol=1e-6)
    logger.info("test_no_target_pattern_match_leaves_data_unchanged: passed")


def test_filter_attenuates_frequency_above_high_cutoff(edf_path):
    logger.info("test_filter_attenuates_frequency_above_high_cutoff: start")
    fs = SFREQ
    t1, t2 = 5, 15
    t1_s, t2_s = int(t1 * fs), int(t2 * fs)

    before = mne.io.read_raw_edf(edf_path, preload=True, verbose=False).get_data()[0, t1_s:t2_s]
    after = bandpass_filter_interval(
        edf_path, t1=t1, t2=t2, target_pattern_fn=only_target, low_cutoff=0.5, high_cutoff=40.0
    ).get_data()[0, t1_s:t2_s]

    freqs, psd_before = signal.welch(before, fs=fs, nperseg=min(1024, len(before)))
    _, psd_after = signal.welch(after, fs=fs, nperseg=min(1024, len(after)))

    idx_60 = np.argmin(np.abs(freqs - 60))
    idx_5 = np.argmin(np.abs(freqs - 5))

    assert psd_after[idx_60] < psd_before[idx_60] * 0.1, (
        f"60 Hz component should be strongly attenuated: before={psd_before[idx_60]:.4f}, "
        f"after={psd_after[idx_60]:.4f}"
    )
    assert psd_after[idx_5] > psd_before[idx_5] * 0.5, (
        "5 Hz passband component should be largely preserved"
    )
    logger.info("test_filter_attenuates_frequency_above_high_cutoff: passed")


@pytest.mark.parametrize(
    "low_cutoff,high_cutoff",
    [
        (0.0, 40.0),   # low_cutoff must be > 0
        (-1.0, 40.0),  # low_cutoff must be > 0
        (0.5, 200.0),  # high_cutoff must be < Nyquist (128 Hz for 256 Hz sfreq)
        (10.0, 5.0),   # low_cutoff must be < high_cutoff
    ],
)
def test_invalid_cutoffs_raise_value_error(edf_path, low_cutoff, high_cutoff):
    logger.info("test_invalid_cutoffs_raise_value_error: start")
    with pytest.raises(ValueError):
        bandpass_filter_interval(
            edf_path, t1=5, t2=15, target_pattern_fn=only_target,
            low_cutoff=low_cutoff, high_cutoff=high_cutoff,
        )
    logger.info("test_invalid_cutoffs_raise_value_error: passed")