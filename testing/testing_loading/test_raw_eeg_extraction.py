import numpy as np
import pytest
from pipeline.raw_eeg_extraction import concatenate_session_eeg

FIXTURE_EDF = "testing/fixtures/sample.edf"


def test_single_recording_shape():
    session = {"edf_paths": [FIXTURE_EDF]}
    # Unpack the tuple: (data, offsets)
    result, offsets = concatenate_session_eeg(session)
    # Updated length after 256Hz resampling is 448000
    assert result.shape == (17, 448000)
    assert len(offsets) == 1


def test_single_recording_offsets():
    """Verify that the offsets list correctly records the start and end samples."""
    session = {"edf_paths": [FIXTURE_EDF]}
    result, offsets = concatenate_session_eeg(session)

    # We expect 1 offset dictionary mapping where this EDF landed
    assert len(offsets) == 1
    assert offsets[0]["edf_path"] == FIXTURE_EDF
    assert offsets[0]["start_sample"] == 0
    assert offsets[0]["end_sample"] == 448000


def test_multiple_recordings_concatenated_shape():
    """Reusing the same fixture twice to simulate a 2-recording session."""
    session = {"edf_paths": [FIXTURE_EDF, FIXTURE_EDF]}
    result, offsets = concatenate_session_eeg(session)
    
    assert result.shape == (17, 448000 * 2)
    assert len(offsets) == 2


def test_multiple_recordings_offset_correctness():
    """
    Second recording's data should land at samples [448000:896000],
    and should be identical to the first recording's data (same fixture
    file used twice).
    """
    session = {"edf_paths": [FIXTURE_EDF, FIXTURE_EDF]}
    result, offsets = concatenate_session_eeg(session)

    # Check that data is identical in both halves
    first_half = result[:, :448000]
    second_half = result[:, 448000:]
    assert np.array_equal(first_half, second_half)

    # Check boundaries in offsets dictionary
    assert offsets[0]["end_sample"] == 448000
    assert offsets[1]["start_sample"] == 448000
    assert offsets[1]["end_sample"] == 896000


def test_empty_session_returns_none():
    session = {"edf_paths": []}
    out = concatenate_session_eeg(session)
    # Handle both return types depending on how your edge case is implemented
    if isinstance(out, tuple):
        assert out[0] is None
    else:
        assert out is None


def test_missing_edf_paths_key_returns_none():
    out = concatenate_session_eeg({})
    if isinstance(out, tuple):
        assert out[0] is None
    else:
        assert out is None


def test_save_to_output_dir(tmp_path):
    session = {"edf_paths": [FIXTURE_EDF]}
    result, offsets = concatenate_session_eeg(
        session, session_key="test_session_001", output_dir=str(tmp_path)
    )

    # Updated file name suffix per your new checkpoint_io logic
    out_file = tmp_path / "test_session_001_raw.npy"
    assert out_file.exists()

    loaded = np.load(out_file)
    assert np.array_equal(loaded, result)


def test_output_dir_without_session_key_raises():
    session = {"edf_paths": [FIXTURE_EDF]}
    with pytest.raises(ValueError):
        concatenate_session_eeg(session, output_dir="/tmp/whatever")