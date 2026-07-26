import pytest
from pipeline.recording_info import get_recording_info

FIXTURE_EDF = "testing/fixtures/sample.edf"


def test_get_recording_info_sfreq():
    info = get_recording_info(FIXTURE_EDF)
    assert info["sfreq"] == 250.0


def test_get_recording_info_n_times():
    info = get_recording_info(FIXTURE_EDF)
    assert info["n_times"] == 437500


def test_get_recording_info_channel_count():
    info = get_recording_info(FIXTURE_EDF)
    assert len(info["ch_names"]) == 17


def test_get_recording_info_channels_are_le_variant():
    """This particular fixture file is an LE-reference recording."""
    info = get_recording_info(FIXTURE_EDF)
    assert all(ch.endswith("-LE") for ch in info["ch_names"])


def test_get_recording_info_returns_dict_with_expected_keys():
    info = get_recording_info(FIXTURE_EDF)
    assert set(info.keys()) == {"sfreq", "n_times", "ch_names"}