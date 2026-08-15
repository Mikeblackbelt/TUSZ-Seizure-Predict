import numpy as np
import pandas as pd


from pipeline.segment_npy import (
    build_edf_to_session_map,
    time_to_sample,
    extract_segments,
    TARGET_SFREQ
)

def test_build_edf_to_session_map():
    sessions = {
        "sess_01": {"edf_paths": ["a.edf", "b.edf"]},
        "sess_02": {"edf_paths": ["c.edf"]},
    }
    
    mapping = build_edf_to_session_map(sessions)
    
    assert mapping == {
        "a.edf": "sess_01",
        "b.edf": "sess_01",
        "c.edf": "sess_02",
    }

    offset = {"start_sample": 100, "end_sample": 1000}
    
    # 1.0 second @ 256 Hz = 256 samples offset by 100 -> 356
    assert time_to_sample(t=1.0, offset=offset, target_sfreq=256) == 356


def test_time_to_sample_clipping_bounds():
    offset = {"start_sample": 100, "end_sample": 500}
    
    # Negative time should clip to start_sample
    assert time_to_sample(t=-5.0, offset=offset, target_sfreq=256) == 100
    
    # Time exceeding file length should clip to end_sample
    assert time_to_sample(t=100.0, offset=offset, target_sfreq=256) 

def test_extract_segments_deduplication_and_slicing():

    sessions = {"sess_1": {"edf_paths": ["a.edf"]}}
    mock_array = np.arange(2000).reshape(2, 1000)  # (2 channels, 1000 samples)
    file_offsets = [{"edf_path": "a.edf", "start_sample": 0, "end_sample": 1000}]
    session_data = {"sess_1": (mock_array, file_offsets)}

    master_df = pd.DataFrame([
        {"edf_path": "a.edf", "start_time": 1.0, "stop_time": 2.0, "label": "fnsz", "is_valid": True, "channel": "FP1-F7"},
        {"edf_path": "a.edf", "start_time": 1.0, "stop_time": 2.0, "label": "fnsz", "is_valid": True, "channel": "F7-T3"},
    ])

    results = extract_segments(master_df, sessions, session_data, dedup_channels=True)

    assert len(results) == 1  
    assert results[0]["channels"] == ["FP1-F7", "F7-T3"]
    
    expected_samples = round(1.0 * TARGET_SFREQ)
    assert results[0]["segment"].shape == (2, expected_samples)


def test_extract_segments_filters_invalid_and_missing():
    sessions = {"sess_1": {"edf_paths": ["a.edf"]}}
    session_data = {"sess_1": (np.zeros((2, 1000)), [{"edf_path": "a.edf", "start_sample": 0, "end_sample": 1000}])}

    master_df = pd.DataFrame([
        {"edf_path": "a.edf", "start_time": 0.0, "stop_time": 1.0, "label": "fnsz", "is_valid": False, "channel": "FP1-F7"}, # Invalid
        {"edf_path": "missing.edf", "start_time": 0.0, "stop_time": 1.0, "label": "fnsz", "is_valid": True, "channel": "FP1-F7"}, # Missing EDF
    ])

    results = extract_segments(master_df, sessions, session_data, is_valid_filter=True)

    assert len(results) == 0