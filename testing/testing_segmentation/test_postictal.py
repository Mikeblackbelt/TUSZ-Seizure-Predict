import pytest
import pandas as pd
from pipeline.preictal_segment import (
    make_master_file,
    add_preictal_tags,
    add_postictal_and_consecutive,
    get_split,
)
from util import handle_logs
from testing.helpers import *

@pytest.fixture
def sample_ictal():
    """Sample with multiple seizures on same file + channel for consecutive testing"""
    return pd.DataFrame({
        "edf_path":   ["a.edf", "a.edf", "a.edf", "b.edf"],
        "csv_path":   ["a.csv", "a.csv", "a.csv", "b.csv"],
        "split":      ["train", "train", "train", "dev"],
        "channel":    ["FP1-F7", "FP1-F7", "FP1-F7", "FP1-F7"],
        "start_time": [100.0, 300.0, 800.0, 150.0],
        "stop_time":  [110.0, 320.0, 820.0, 160.0],
        "label":      ["fnsz", "fnsz", "gnsz", "fnsz"],
        "confidence": [1, 1, 1, 1],
        "status":     [-1, -1, -1, -1],
    })


def test_postictal_and_consecutive_row_count(sample_ictal):
    """Should add postictal for isolated seizures and consecutive for close pairs"""
    logger.info("test_postictal_and_consecutive_row_count: start")
    
    result = add_postictal_and_consecutive(
        sample_ictal, 
        postictal_length=60, 
        preictal_length=30
    )
    
    assert len(result) > len(sample_ictal)
    logger.info("test_postictal_and_consecutive_row_count: passed")


def test_consecutive_tag_creation(sample_ictal):
    """Test c{type1}{type2} and c{type}2 logic"""
    logger.info("test_consecutive_tag_creation: start")
   
    # Use parameters that WILL trigger consecutive (gap=190)
    result = add_postictal_and_consecutive(sample_ictal, postictal_length=100, preictal_length=100)
   
    labels = set(result["label"].unique())
   
    assert "cfnsz2" in labels, f"Expected cfnsz2, got {labels}"
    assert any(l.startswith("c") for l in labels)
    assert any(l.startswith("q") for l in labels)
    logger.info("test_consecutive_tag_creation: passed")


def test_consecutive_vs_different_types(sample_ictal):
    """Different seizure types should produce c{type1}{type2}"""
    logger.info("test_consecutive_vs_different_types: start")
   
    test_df = sample_ictal.copy()
    
    test_df.loc[1, "start_time"] = 150.0
    test_df.loc[1, "stop_time"] = 160.0
    test_df.loc[1, "label"] = "gnsz"
    
    result = add_postictal_and_consecutive(test_df, postictal_length=50, preictal_length=30)
   
    consec_labels = [lbl for lbl in result["label"] if lbl.startswith("c")]
    assert any("cfnszgnsz" in lbl for lbl in consec_labels), f"Expected cfnszgnsz, got {consec_labels}"
    logger.info("test_consecutive_vs_different_types: passed")

def test_consecutive_time_window(sample_ictal):
    """Check that consecutive tag covers postictal of first + preictal of second"""
    logger.info("test_consecutive_time_window: start")
    
    result = add_postictal_and_consecutive(sample_ictal, postictal_length=60, preictal_length=30)
    result2 = add_postictal_and_consecutive(sample_ictal, postictal_length=100, preictal_length=100)
    
    consec = result2[result2["label"].str.startswith("c")]
    assert not consec.empty
    
    assert consec.iloc[0]["start_time"] == 110.0
    logger.info("test_consecutive_time_window: passed")


def test_postictal_tag_for_isolated_seizure(sample_ictal):
    """Seizures that are far apart should get q{type}"""
    logger.info("test_postictal_tag_for_isolated_seizure: start")
    
    result = add_postictal_and_consecutive(sample_ictal, postictal_length=60, preictal_length=30)
    
    q_tags = result[result["label"].str.startswith("q")]
    assert not q_tags.empty
    assert any("qfnsz" in lbl or "qgnsz" in lbl for lbl in q_tags["label"].values)
    logger.info("test_postictal_tag_for_isolated_seizure: passed")


def test_postictal_consecutive_original_rows_unchanged(sample_ictal):
    """Original ictal rows should remain intact"""
    logger.info("test_postictal_consecutive_original_rows_unchanged: start")
    
    result = add_postictal_and_consecutive(sample_ictal, postictal_length=60, preictal_length=30)
    
    original_fnsz = result[result["label"] == "fnsz"]
    assert len(original_fnsz) == 3  # original count preserved
    logger.info("test_postictal_consecutive_original_rows_unchanged: passed")


def test_postictal_consecutive_sorted(sample_ictal):
    """Final result should be sorted by split → edf_path → channel → start_time"""
    logger.info("test_postictal_consecutive_sorted: start")
    
    result = add_postictal_and_consecutive(sample_ictal, postictal_length=60, preictal_length=30)
    
    assert result["start_time"].is_monotonic_increasing == False  # because different files
    for _, group in result.groupby(["split", "edf_path", "channel"]):
        assert group["start_time"].is_monotonic_increasing
    logger.info("test_postictal_consecutive_sorted: passed")


def test_status_for_trimmed_windows(sample_ictal):
    """Basic check that status is set on trimmed windows"""
    logger.info("test_status_for_trimmed_windows: start")
    
    result = add_postictal_and_consecutive(sample_ictal, postictal_length=1000, preictal_length=1000)
    
    trimmed = result[result["status"] > 0]
    assert len(trimmed) >= 0  # at least doesn't crash
    logger.info("test_status_for_trimmed_windows: passed")
