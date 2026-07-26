import numpy as np
import pytest
from pipeline.raw_eeg_extraction import concatenate_session_eeg

FIXTURE_EDF = "testing/fixtures/sample.edf"


def test_single_recording_shape():
    session = {"edf_paths": [FIXTURE_EDF]}
    result = concatenate_session_eeg(session)
    assert result.shape == (17, 437500)


def test_single_recording_matches_direct_read():
    """Sanity check: concatenating one file should equal reading it directly."""
    import mne
    from pipeline.eeg_channels import CHANNELS_TO_INCLUDE

    session = {"edf_paths": [FIXTURE_EDF]}
    result = concatenate_session_eeg(session)

    raw = mne.io.read_raw_edf(FIXTURE_EDF, include=CHANNELS_TO_INCLUDE, verbose="Warning")
    direct = raw.get_data()

    assert np.array_equal(result, direct)


def test_multiple_recordings_concatenated_shape():
    """Reusing the same fixture twice to simulate a 2-recording session."""
    session = {"edf_paths": [FIXTURE_EDF, FIXTURE_EDF]}
    result = concatenate_session_eeg(session)
    assert result.shape == (17, 437500 * 2)


def test_multiple_recordings_offset_correctness():
    """
    Second recording's data should land at samples [437500:875000],
    and should be identical to the first recording's data (same fixture
    file used twice).
    """
    session = {"edf_paths": [FIXTURE_EDF, FIXTURE_EDF]}
    result = concatenate_session_eeg(session)

    first_half = result[:, :437500]
    second_half = result[:, 437500:]
    assert np.array_equal(first_half, second_half)


def test_empty_session_returns_none():
    session = {"edf_paths": []}
    result = concatenate_session_eeg(session)
    assert result is None


def test_missing_edf_paths_key_returns_none():
    result = concatenate_session_eeg({})
    assert result is None


def test_save_to_output_dir(tmp_path):
    session = {"edf_paths": [FIXTURE_EDF]}
    result = concatenate_session_eeg(
        session, session_key="test_session_001", output_dir=str(tmp_path)
    )

    out_file = tmp_path / "test_session_001.npy"
    assert out_file.exists()

    loaded = np.load(out_file)
    assert np.array_equal(loaded, result)


def test_output_dir_without_session_key_raises():
    session = {"edf_paths": [FIXTURE_EDF]}
    with pytest.raises(ValueError):
        concatenate_session_eeg(session, output_dir="/tmp/whatever")