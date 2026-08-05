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
    """
    Two isolated seizures, far enough from session start and from each
    other that both should clear Gate 1 and Gate 2 under the default
    sph/sop/postictal_time values used below (sph=10, sop=50 -> need
    start_time >= 60; gap needs postictal+sph+sop, satisfied by the
    300s spacing between the two seizures).
    """
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
    result = add_preictal_tags(sample_master, sph=10, sop=50)
    assert len(result) == len(sample_master) * 3
    logger.info("test_preictal_row_count: passed")


def test_preictal_labels(sample_master):
    logger.info("test_preictal_labels: start")
    result = add_preictal_tags(sample_master, sph=10, sop=50)
    assert "pfnsz" in result["label"].values
    assert "pgnsz" in result["label"].values
    logger.info("test_preictal_labels: passed")


def test_preictal_times_full_window(sample_master):
    # row1: start=100, sph=10 (window), sop=50 (buffer) -> window = [100-50-10, 100-50] = [40, 50]
    #   Gate 1: 100 >= sph+sop (60) -> passes. No previous seizure -> Gate 2 n/a.
    # row2: start=400, sph=10, sop=50 -> window = [400-50-10, 400-50] = [340, 350]
    #   Gate 1: 400 >= 60 -> passes. Different (edf_path, channel) group from
    #   row1's "a.edf" isn't true here (same edf_path/channel) but different
    #   split - grouping is by (edf_path, channel) only, so row2 IS treated
    #   as the seizure after row1 in the same group. Gap = 400 - 110 = 290.
    #   postictal_time not passed here (None) -> Gate 2 skipped entirely.
    logger.info("test_preictal_times_full_window: start")
    result = add_preictal_tags(sample_master, sph=10, sop=50)

    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 40.0
    assert pfnsz["stop_time"] == 50.0
    assert pfnsz["status"] == 1

    pgnsz = result[result["label"] == "pgnsz"].iloc[0]
    assert pgnsz["start_time"] == 340.0
    assert pgnsz["stop_time"] == 350.0
    assert pgnsz["status"] == 1
    logger.info("test_preictal_times_full_window: passed")


def test_preictal_gate1_session_start_failure(sample_master):
    """
    Gate 1: if start_time < sph + sop, the window can't fit before session
    start at all - dropped entirely (status=0), not trimmed.
    """
    logger.info("test_preictal_gate1_session_start_failure: start")
    # row1: start=100, sph=10, sop=200 -> sph+sop=210 > 100 -> Gate 1 fails
    result = add_preictal_tags(sample_master, sph=10, sop=200)
    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 0.0
    assert pfnsz["stop_time"] == 0.0
    assert pfnsz["status"] == 0
    logger.info("test_preictal_gate1_session_start_failure: passed")

def test_preictal_gate1_session_start_failure(sample_master):
    """
    Gate 1: if start_time < sph + sop, the window can't fit before session
    start at all - dropped, but kept as a real-width row [0, j_start] so
    downstream interictal logic correctly treats it as occupied time.
    """
    logger.info("test_preictal_gate1_session_start_failure: start")
    # row1: start=100, sph=10, sop=200 -> sph+sop=210 > 100 -> Gate 1 fails
    result = add_preictal_tags(sample_master, sph=10, sop=200)
    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 0.0
    assert pfnsz["stop_time"] == 100.0  # == j_start
    assert pfnsz["status"] == 0
    logger.info("test_preictal_gate1_session_start_failure: passed")


def test_preictal_gate1_uses_full_sph_plus_sop(sample_master):
    """
    Gate 1 fails as soon as sph+sop alone exceeds start_time, regardless of
    postictal_time (Gate 1 doesn't involve postictal_time at all).
    """
    logger.info("test_preictal_gate1_uses_full_sph_plus_sop: start")
    # row1: start=100, sph=150, sop=50 -> sph+sop=200 > 100 -> Gate 1 fails
    result = add_preictal_tags(sample_master, sph=150, sop=50)
    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["status"] == 0
    assert pfnsz["start_time"] == 0.0
    assert pfnsz["stop_time"] == 100.0  # == j_start
    logger.info("test_preictal_gate1_uses_full_sph_plus_sop: passed")


def test_preictal_gate2_inter_seizure_gap_failure():
    """
    Gate 2: if the gap between the previous seizure's end and this
    seizure's start is smaller than postictal_time + sph + sop, this
    seizure's preictal window is dropped, but kept as a real-width row
    [i_end, j_start] so downstream interictal logic treats that gap as
    occupied rather than free background.
    """
    logger.info("test_preictal_gate2_inter_seizure_gap_failure: start")
    df = pd.DataFrame({
        "edf_path":   ["a.edf", "a.edf"],
        "csv_path":   ["a.csv", "a.csv"],
        "split":      ["train", "train"],
        "channel":    ["FP1-F7", "FP1-F7"],
        "start_time": [100.0, 200.0],   # gap from seizure1's end (110) = 90
        "stop_time":  [110.0, 210.0],
        "label":      ["fnsz", "fnsz"],
        "confidence": [1, 1],
        "status":     [-1, -1],
    })
  
    result = add_preictal_tags(df, sph=10, sop=20, postictal_time=100)
    pfnsz = result[result["label"] == "pfnsz"].sort_values("start_time").reset_index(drop=True)

    # seizure1 (i=0): no previous seizure, own start_time=100 clears Gate 1
    # (sph+sop=30) -> viable, status=1
    assert pfnsz.iloc[0]["status"] == 1

    # seizure2 (i=1): fails Gate 2 -> dropped, real-width row [i_end=110, j_start=200]
    assert pfnsz.iloc[1]["status"] == 0
    assert pfnsz.iloc[1]["start_time"] == 110.0
    assert pfnsz.iloc[1]["stop_time"] == 200.0
    logger.info("test_preictal_gate2_inter_seizure_gap_failure: passed")

def test_preictal_gate2_skipped_when_postictal_time_none():
    """
    Gate 2 should be skipped entirely (only Gate 1 applies) when
    postictal_time is not provided, even for seizures close together.
    """
    logger.info("test_preictal_gate2_skipped_when_postictal_time_none: start")
    df = pd.DataFrame({
        "edf_path":   ["a.edf", "a.edf"],
        "csv_path":   ["a.csv", "a.csv"],
        "split":      ["train", "train"],
        "channel":    ["FP1-F7", "FP1-F7"],
        "start_time": [100.0, 200.0],
        "stop_time":  [110.0, 210.0],
        "label":      ["fnsz", "fnsz"],
        "confidence": [1, 1],
        "status":     [-1, -1],
    })
    # sph=10, sop=20 -> second seizure's window = [200-20-10, 200-20] = [170, 180]
    # Gate 1: 200 >= 30 -> passes. postictal_time=None -> Gate 2 not evaluated.
    result = add_preictal_tags(df, sph=10, sop=20, postictal_time=None)
    pfnsz_rows = result[result["label"] == "pfnsz"].sort_values("start_time").reset_index(drop=True)
    assert pfnsz_rows.iloc[1]["status"] == 1
    assert pfnsz_rows.iloc[1]["start_time"] == 170.0
    assert pfnsz_rows.iloc[1]["stop_time"] == 180.0
    logger.info("test_preictal_gate2_skipped_when_postictal_time_none: passed")

def test_preictal_no_partial_windows_ever(sample_master):
    logger.info("test_preictal_no_partial_windows_ever: start")
    result = add_preictal_tags(sample_master, sph=10, sop=50)
    is_sopbuffer = result["label"].str.endswith("_sopbuffer")
    preictal_rows = result[result["label"].str.startswith("p") & ~is_sopbuffer]
    sopbuffer_rows = result[is_sopbuffer]

    valid = preictal_rows[preictal_rows["status"] == 1]
    lengths = (valid["stop_time"] - valid["start_time"]).round(6)
    assert (lengths == 10.0).all()  # sph (extracted window duration)

    dropped = preictal_rows[preictal_rows["status"] == 0]
    assert (dropped["stop_time"] >= dropped["start_time"]).all()  # real-width, gate1/gate2 dropped rows

    sopbuffer_lengths = (sopbuffer_rows["stop_time"] - sopbuffer_rows["start_time"]).round(6)
    assert (sopbuffer_lengths == 50.0).all()  # sop
    assert (sopbuffer_rows["status"] == 1).all()
    logger.info("test_preictal_no_partial_windows_ever: passed")

def test_preictal_status_never_negative(sample_master):
    logger.info("test_preictal_status_never_negative: start")
    result = add_preictal_tags(sample_master, sph=9999, sop=9999)
    preictal_rows = result[result["label"].str.startswith("p")]
    assert (preictal_rows["start_time"] >= 0).all()
    assert (preictal_rows["stop_time"] >= 0).all()
    logger.info("test_preictal_status_never_negative: passed")


def test_preictal_original_rows_unchanged(sample_master):
    logger.info("test_preictal_original_rows_unchanged: start")
    result = add_preictal_tags(sample_master, sph=10, sop=50)
    fnsz = result[result["label"] == "fnsz"].iloc[0]
    assert fnsz["start_time"] == 100.0
    assert fnsz["stop_time"] == 110.0
    assert fnsz["status"] == -1
    logger.info("test_preictal_original_rows_unchanged: passed")


def test_preictal_sorted_by_split_then_time(sample_master):
    logger.info("test_preictal_sorted_by_split_then_time: start")
    result = add_preictal_tags(sample_master, sph=10, sop=50)
    for split, group in result.groupby("split"):
        times = group["start_time"].tolist()
        assert times == sorted(times)
    logger.info("test_preictal_sorted_by_split_then_time: passed")