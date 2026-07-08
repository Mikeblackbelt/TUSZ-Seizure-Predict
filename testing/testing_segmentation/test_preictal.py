import pytest
import pandas as pd
from pipeline.preictal_segment import (
    get_unique_tags,
    make_master_file,
    add_preictal_tags,
    get_split,
)
from util import handle_logs
from testing.helpers import *


@pytest.fixture
def sample_master():
    return pd.DataFrame({
        "edf_path":   ["a.edf", "a.edf"],
        "csv_path":   ["a.csv", "a.csv"],
        "split":      ["train", "dev"],
        "channel":    ["FP1-F7", "FP1-F7"],
        "start_time": [100.0, 400.0],
        "stop_time":  [110.0, 420.0],
        "label":      ["fnsz", "gnsz"],
        "confidence": [1, 1],
        "status":     [-1, -1],
    })

def test_preictal_row_count(sample_master):
    logger.info("test_preictal_row_count: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    assert len(result) == len(sample_master) * 2
    logger.info("test_preictal_row_count: passed")

def test_preictal_labels(sample_master):
    logger.info("test_preictal_labels: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    assert "pfnsz" in result["label"].values
    assert "pgnsz" in result["label"].values
    logger.info("test_preictal_labels: passed")

def test_preictal_times_clean(sample_master):
    # row1: start=100, cutoff=10, dur=50 -> raw_end=90, raw_start=40 -> both > 0
    # row2: start=400, cutoff=10, dur=50 -> raw_end=390, raw_start=340 -> both > 0
    logger.info("test_preictal_times_clean: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)

    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 40.0
    assert pfnsz["stop_time"] == 90.0
    assert pfnsz["status"] == 0

    pgnsz = result[result["label"] == "pgnsz"].iloc[0]
    assert pgnsz["start_time"] == 340.0
    assert pgnsz["stop_time"] == 390.0
    assert pgnsz["status"] == 0
    logger.info("test_preictal_times_clean: passed")

def test_preictal_status_start_trimmed(sample_master):
    # row1: start=100, cutoff=10 -> raw_end=90 (>0), dur=200 -> raw_start=-110 (<=0)
    # expect status=1, start clamped to 0, stop stays at raw_end
    logger.info("test_preictal_status_start_trimmed: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=200)
    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 0.0
    assert pfnsz["stop_time"] == 90.0
    assert pfnsz["status"] == 1
    logger.info("test_preictal_status_start_trimmed: passed")

def test_preictal_status_collapsed(sample_master):
    # row1: start=100, cutoff=150 -> raw_end=-50 (<=0)
    # expect status=2, start and stop both clamped to 0
    logger.info("test_preictal_status_collapsed: start")
    result = add_preictal_tags(sample_master, start_cutoff=150, max_duration=50)
    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 0.0
    assert pfnsz["stop_time"] == 0.0
    assert pfnsz["status"] == 2
    logger.info("test_preictal_status_collapsed: passed")

def test_preictal_status_never_negative(sample_master):
    logger.info("test_preictal_status_never_negative: start")
    result = add_preictal_tags(sample_master, start_cutoff=9999, max_duration=9999)
    preictal_rows = result[result["label"].str.startswith("p")]
    assert (preictal_rows["start_time"] >= 0).all()
    assert (preictal_rows["stop_time"] >= 0).all()
    logger.info("test_preictal_status_never_negative: passed")

def test_preictal_original_rows_unchanged(sample_master):
    logger.info("test_preictal_original_rows_unchanged: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    fnsz = result[result["label"] == "fnsz"].iloc[0]
    assert fnsz["start_time"] == 100.0
    assert fnsz["stop_time"] == 110.0
    assert fnsz["status"] == -1
    logger.info("test_preictal_original_rows_unchanged: passed")

def test_preictal_sorted_by_split_then_time(sample_master):
    logger.info("test_preictal_sorted_by_split_then_time: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    # within each split group, start_time should be non-decreasing
    for split, group in result.groupby("split"):
        times = group["start_time"].tolist()
        assert times == sorted(times)
    logger.info("test_preictal_sorted_by_split_then_time: passed")