import pytest
import pandas as pd
from pipeline.preictal_segment import (
    make_master_file,
    add_preictal_tags,
    add_exclusion_intervals,
    get_split,
)
from util import handle_logs
from testing.helpers import *

@pytest.fixture
def sample_ictal():
    """Sample with multiple seizures on same file + channel for testing exclusion intervals"""
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
    """Should add exclusion intervals for seizures"""
    logger.info("test_postictal_and_consecutive_row_count: start")
    
    result = add_exclusion_intervals(
        master_df=sample_ictal, 
        postictal_time=60
    )
    assert len(result) > len(sample_ictal)
    logger.info("test_postictal_and_consecutive_row_count: passed")


def test_consecutive_tag_creation(sample_ictal):
    """Test x{type} exclusion tag creation logic"""
    logger.info("test_consecutive_tag_creation: start")
   
    result = add_exclusion_intervals(
        master_df=sample_ictal,
        postictal_time=100
    )
   
    labels = set(result["label"].unique())
   
    assert "xfnsz" in labels, f"Expected xfnsz in labels, got {labels}"
    assert any(l.startswith("x") for l in labels)
    logger.info("test_consecutive_tag_creation: passed")


def test_consecutive_vs_different_types(sample_ictal):
    """Different seizure types should produce x{type1} and x{type2} exclusion windows"""
    logger.info("test_consecutive_vs_different_types: start")
   
    test_df = sample_ictal.copy()
    
    test_df.loc[1, "start_time"] = 150.0
    test_df.loc[1, "stop_time"] = 160.0
    test_df.loc[1, "label"] = "gnsz"
    
    result = add_exclusion_intervals(
        master_df=test_df,
        postictal_time=50
    )
   
    exclusion_labels = [lbl for lbl in result["label"] if lbl.startswith("x")]
    assert "xfnsz" in exclusion_labels and "xgnsz" in exclusion_labels, f"Expected xfnsz and xgnsz, got {exclusion_labels}"
    logger.info("test_consecutive_vs_different_types: passed")


def test_consecutive_time_window(sample_ictal):
    """Check that exclusion interval start time aligns with seizure stop time"""
    logger.info("test_consecutive_time_window: start")
    
    result = add_exclusion_intervals(
        master_df=sample_ictal,
        postictal_time=100
    )
    
    exclusions = result[result["label"].str.startswith("x")]
    assert not exclusions.empty
    
    # Filter for 'a.edf' specifically (since 'dev' split sorts before 'train')
    a_excl = exclusions[exclusions["edf_path"] == "a.edf"]
    assert a_excl.iloc[0]["start_time"] == 110.0
    logger.info("test_consecutive_time_window: passed")


def test_postictal_tag_for_isolated_seizure(sample_ictal):
    """Seizures should generate x{type} exclusion tags"""
    logger.info("test_postictal_tag_for_isolated_seizure: start")
    
    result = add_exclusion_intervals(
        master_df=sample_ictal,
        postictal_time=60
    )
    
    x_tags = result[result["label"].str.startswith("x")]
    assert not x_tags.empty
    assert any("xfnsz" in lbl or "xgnsz" in lbl for lbl in x_tags["label"].values)
    logger.info("test_postictal_tag_for_isolated_seizure: passed")


def test_postictal_consecutive_original_rows_unchanged(sample_ictal):
    """Original ictal rows should remain intact"""
    logger.info("test_postictal_consecutive_original_rows_unchanged: start")
    
    result = add_exclusion_intervals(
        master_df=sample_ictal,
        postictal_time=60
    )
    
    original_fnsz = result[result["label"] == "fnsz"]
    assert len(original_fnsz) == 3  # original count preserved
    logger.info("test_postictal_consecutive_original_rows_unchanged: passed")


def test_postictal_consecutive_sorted(sample_ictal):
    """Final result should be sorted by split -> edf_path -> channel -> start_time"""
    logger.info("test_postictal_consecutive_sorted: start")
    
    result = add_exclusion_intervals(
        master_df=sample_ictal,
        postictal_time=60
    )
    
    assert result["start_time"].is_monotonic_increasing == False  # because different files
    for _, group in result.groupby(["split", "edf_path", "channel"]):
        assert group["start_time"].is_monotonic_increasing
    logger.info("test_postictal_consecutive_sorted: passed")


def test_status_for_exclusion_windows(sample_ictal):
    """Basic check that status is handled on exclusion windows"""
    logger.info("test_status_for_exclusion_windows: start")
    
    result = add_exclusion_intervals(
        master_df=sample_ictal,
        postictal_time=1000
    )
    
    exclusion_rows = result[result["label"].str.startswith("x")]
    assert not exclusion_rows.empty
    logger.info("test_status_for_exclusion_windows: passed")