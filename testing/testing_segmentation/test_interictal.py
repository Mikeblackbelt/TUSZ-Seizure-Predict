import pandas as pd
import pytest
import logging

from pipeline.preictal_segment import add_interictal_tags

logger = logging.getLogger(__name__)


@pytest.fixture
def base_row():
    return {
        "edf_path": "train/a/s001/a_s001_t000.edf",
        "csv_path": "train/a/s001/a_s001_t000.csv",
        "split": "train",
        "channel": "C3-CZ",
        "confidence": 1.0,
    }


def make_row(base_row, start, stop, label, is_valid):
    return {**base_row, "start_time": start, "stop_time": stop,
            "label": label, "is_valid": is_valid}


def test_interictal_fills_single_middle_gap(base_row):
    logger.info("test_interictal_fills_single_middle_gap: start")
    master_df = pd.DataFrame([
        make_row(base_row, 0.0, 100.0, "gnsz", -1),
        make_row(base_row, 500.0, 600.0, "xgnsz", 1),
    ])
    durations = {base_row["edf_path"]: 1000.0}

    result = add_interictal_tags(master_df, durations, min_interictal_length=0)
    interictal = result[result["label"] == "interictal"].sort_values("start_time")

    assert len(interictal) == 2
    assert (interictal.iloc[0]["start_time"], interictal.iloc[0]["stop_time"]) == (100.0, 500.0)
    assert (interictal.iloc[1]["start_time"], interictal.iloc[1]["stop_time"]) == (600.0, 1000.0)
    logger.info("test_interictal_fills_single_middle_gap: passed")


def test_interictal_is_valid_always_true(base_row):
    logger.info("test_interictal_is_valid_always_true: start")
    master_df = pd.DataFrame([make_row(base_row, 0.0, 100.0, "gnsz", True)])
    durations = {base_row["edf_path"]: 500.0}
 
    result = add_interictal_tags(master_df, durations)
    interictal = result[result["label"] == "interictal"]
 
    assert len(interictal) > 0
    assert (interictal["is_valid"] == True).all()
    logger.info("test_interictal_is_valid_always_true: passed")


def test_interictal_leading_gap_from_zero(base_row):
    logger.info("test_interictal_leading_gap_from_zero: start")
    master_df = pd.DataFrame([make_row(base_row, 200.0, 300.0, "gnsz", -1)])
    durations = {base_row["edf_path"]: 300.0}

    result = add_interictal_tags(master_df, durations)
    interictal = result[result["label"] == "interictal"]

    assert len(interictal) == 1
    assert interictal.iloc[0]["start_time"] == 0.0
    assert interictal.iloc[0]["stop_time"] == 200.0
    logger.info("test_interictal_leading_gap_from_zero: passed")


def test_interictal_no_trailing_gap_when_flush_with_duration(base_row):
    logger.info("test_interictal_no_trailing_gap_when_flush_with_duration: start")
    master_df = pd.DataFrame([make_row(base_row, 0.0, 1000.0, "gnsz", -1)])
    durations = {base_row["edf_path"]: 1000.0}

    result = add_interictal_tags(master_df, durations)
    assert (result["label"] == "interictal").sum() == 0
    logger.info("test_interictal_no_trailing_gap_when_flush_with_duration: passed")


def test_interictal_short_gap_dropped_below_min_length(base_row):
    logger.info("test_interictal_short_gap_dropped_below_min_length: start")
    master_df = pd.DataFrame([
        make_row(base_row, 0.0, 100.0, "gnsz", -1),
        make_row(base_row, 105.0, 200.0, "xgnsz", 1),  # 5s gap
    ])
    durations = {base_row["edf_path"]: 200.0}

    result = add_interictal_tags(master_df, durations, min_interictal_length=10)
    interictal = result[result["label"] == "interictal"]

    assert not ((interictal["start_time"] == 100.0) & (interictal["stop_time"] == 105.0)).any()
    logger.info("test_interictal_short_gap_dropped_below_min_length: passed")


def test_interictal_gap_exactly_at_min_length_is_kept(base_row):
    logger.info("test_interictal_gap_exactly_at_min_length_is_kept: start")
    master_df = pd.DataFrame([
        make_row(base_row, 0.0, 100.0, "gnsz", -1),
        make_row(base_row, 110.0, 200.0, "xgnsz", 1),  # exactly 10s gap
    ])
    durations = {base_row["edf_path"]: 200.0}

    result = add_interictal_tags(master_df, durations, min_interictal_length=10)
    interictal = result[result["label"] == "interictal"]

    assert ((interictal["start_time"] == 100.0) & (interictal["stop_time"] == 110.0)).any()
    logger.info("test_interictal_gap_exactly_at_min_length_is_kept: passed")


def test_interictal_overlapping_rows_merged_before_gap_detection(base_row):
    logger.info("test_interictal_overlapping_rows_merged_before_gap_detection: start")
    master_df = pd.DataFrame([
        make_row(base_row, 100.0, 300.0, "xgnsz", 1),
        make_row(base_row, 0.0, 150.0, "gnsz", -1),   # overlaps the row above
        make_row(base_row, 400.0, 500.0, "xfnsz", 1),
    ])
    durations = {base_row["edf_path"]: 500.0}

    result = add_interictal_tags(master_df, durations)
    interictal = result[result["label"] == "interictal"]

    assert len(interictal) == 1
    assert interictal.iloc[0]["start_time"] == 300.0
    assert interictal.iloc[0]["stop_time"] == 400.0
    logger.info("test_interictal_overlapping_rows_merged_before_gap_detection: passed")


def test_interictal_dropped_preictal_rows_dont_block_gap_filling(base_row):
    logger.info("test_interictal_dropped_preictal_rows_dont_block_gap_filling: start")
    # pfnsz here is a Gate-1-failed row (is_valid=False) - a rejected candidate
    # window, not real labeled time. It must not count as "covered", or a
    # real stretch of background activity would be wrongly skipped.
    master_df = pd.DataFrame([
        make_row(base_row, 0.0, 50.0, "pfnsz", 0),
        make_row(base_row, 50.0, 80.0, "fnsz", -1),
        make_row(base_row, 80.0, 1880.0, "xfnsz", 1),
    ])
    durations = {base_row["edf_path"]: 2000.0}

    result = add_interictal_tags(master_df, durations)
    interictal = result[result["label"] == "interictal"]

    assert len(interictal) == 1
    assert interictal.iloc[0]["start_time"] == 1880.0
    assert interictal.iloc[0]["stop_time"] == 2000.0
    logger.info("test_interictal_dropped_preictal_rows_dont_block_gap_filling: passed")


def test_interictal_multiple_channels_independent(base_row):
    logger.info("test_interictal_multiple_channels_independent: start")
    master_df = pd.DataFrame([
        make_row(base_row, 0.0, 100.0, "gnsz", -1),
        {**make_row(base_row, 0.0, 900.0, "gnsz", -1), "channel": "C4-P4"},
    ])
    durations = {base_row["edf_path"]: 1000.0}

    result = add_interictal_tags(master_df, durations)
    c3 = result[(result["channel"] == "C3-CZ") & (result["label"] == "interictal")]
    c4 = result[(result["channel"] == "C4-P4") & (result["label"] == "interictal")]

    assert len(c3) == 1 and c3.iloc[0]["start_time"] == 100.0
    assert len(c4) == 1 and c4.iloc[0]["start_time"] == 900.0
    logger.info("test_interictal_multiple_channels_independent: passed")