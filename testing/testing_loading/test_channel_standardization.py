"""Unit tests for the channel standardization helpers in pipeline.process_signal:
    standardize_channel_name, standardize_channels_names, drop_channels
"""
import mne
import numpy as np
import pandas as pd
import pytest

from pipeline.process_signal import (
    standard_channels,
    standardize_channel_name,
    standardize_channels_names,
    drop_channels,
)
from util import handle_logs

logger = handle_logs.get_logger("test_channel_standardization", "logs/test.log")

SFREQ = 256.0
N_SAMPLES = 100


def make_raw_and_metadata(ch_names, path="dummy.edf"):
    """Build a synthetic Raw object plus a metadata DataFrame matching the
    shape produced by pipeline.process_signal.load_edf."""
    info = mne.create_info(ch_names, SFREQ, ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(np.zeros((len(ch_names), N_SAMPLES)), info, verbose=False)
    metadata = pd.DataFrame({
        "path": [path],
        "channels": [list(ch_names)],
        "sfreq": [SFREQ],
        "n_samples": [N_SAMPLES],
        "duration_sec": [N_SAMPLES / SFREQ],
    })
    return raw, metadata


def test_standardize_channel_name_strips_ref_suffix():
    logger.info("test_standardize_channel_name_strips_ref_suffix: start")
    assert standardize_channel_name("EEG FP1-REF") == "FP1"
    logger.info("test_standardize_channel_name_strips_ref_suffix: passed")


def test_standardize_channel_name_strips_le_suffix():
    logger.info("test_standardize_channel_name_strips_le_suffix: start")
    assert standardize_channel_name("EEG FP1-LE") == "FP1"
    logger.info("test_standardize_channel_name_strips_le_suffix: passed")


def test_standardize_channel_name_returns_none_for_non_eeg_channel():
    logger.info("test_standardize_channel_name_returns_none_for_non_eeg_channel: start")
    assert standardize_channel_name("EKG1-REF") is None
    assert standardize_channel_name("BURSTS") is None
    logger.info("test_standardize_channel_name_returns_none_for_non_eeg_channel: passed")


def test_standardize_channel_name_removes_internal_spaces():
    logger.info("test_standardize_channel_name_removes_internal_spaces: start")
    assert standardize_channel_name("EEG T3-REF") == "T3"
    logger.info("test_standardize_channel_name_removes_internal_spaces: passed")


def test_standardize_channels_names_keeps_only_known_standard_channels():
    logger.info("test_standardize_channels_names_keeps_only_known_standard_channels: start")
    ch_names = ["EEG FP1-REF", "EEG F7-REF", "EKG1-REF", "EEG 31-REF"]
    raw, metadata = make_raw_and_metadata(ch_names)

    new_raw, new_metadata = standardize_channels_names(raw, metadata)

    assert set(new_raw.ch_names) == {"FP1", "F7"}
    assert set(new_metadata["channels"].iloc[0]) == {"FP1", "F7"}
    logger.info("test_standardize_channels_names_keeps_only_known_standard_channels: passed")


def test_standardize_channels_names_renames_in_place_on_raw():
    logger.info("test_standardize_channels_names_renames_in_place_on_raw: start")
    ch_names = ["EEG FP1-REF", "EEG FP2-LE"]
    raw, metadata = make_raw_and_metadata(ch_names)

    new_raw, _ = standardize_channels_names(raw, metadata)

    assert "FP1" in new_raw.ch_names
    assert "FP2" in new_raw.ch_names
    assert "EEG FP1-REF" not in new_raw.ch_names
    logger.info("test_standardize_channels_names_renames_in_place_on_raw: passed")


def test_drop_channels_returns_none_when_channels_missing():
    logger.info("test_drop_channels_returns_none_when_channels_missing: start")
    # Only provide a handful of the 19 required standard channels.
    ch_names = [f"EEG {ch}-REF" for ch in standard_channels[:5]]
    raw, metadata = make_raw_and_metadata(ch_names)

    result = drop_channels(raw, metadata)

    assert result is None
    logger.info("test_drop_channels_returns_none_when_channels_missing: passed")


def test_drop_channels_reorders_channels_to_desired_order():
    logger.info("test_drop_channels_reorders_channels_to_desired_order: start")
    shuffled = list(reversed(standard_channels))
    ch_names = [f"EEG {ch}-REF" for ch in shuffled]
    raw, metadata = make_raw_and_metadata(ch_names)

    result = drop_channels(raw, metadata)

    assert result is not None
    assert result.ch_names == standard_channels
    logger.info("test_drop_channels_reorders_channels_to_desired_order: passed")


def test_drop_channels_drops_extra_non_standard_channels():
    logger.info("test_drop_channels_drops_extra_non_standard_channels: start")
    ch_names = [f"EEG {ch}-REF" for ch in standard_channels] + ["EEG 31-REF", "EKG1-REF"]
    raw, metadata = make_raw_and_metadata(ch_names)

    result = drop_channels(raw, metadata)

    assert result is not None
    assert result.ch_names == standard_channels
    assert metadata["channels"].iloc[0] == standard_channels
    logger.info("test_drop_channels_drops_extra_non_standard_channels: passed")


def test_drop_channels_accepts_custom_desired_order():
    logger.info("test_drop_channels_accepts_custom_desired_order: start")
    subset = standard_channels[:3]
    ch_names = [f"EEG {ch}-REF" for ch in reversed(subset)]
    raw, metadata = make_raw_and_metadata(ch_names)

    result = drop_channels(raw, metadata, desired_order=subset)

    assert result is not None
    assert result.ch_names == subset
    logger.info("test_drop_channels_accepts_custom_desired_order: passed")