import pytest
import pandas as pd
from pipeline.preictal_segment import (
    make_master_file,
)
from util import handle_logs
from testing.helpers import *
logger = handle_logs.get_logger("test_pipeline", "applog")

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
        "is_valid":     [True, True],
    })

def test_make_master_file_basic(dataset_dir):
    logger.info("test_make_master_file_basic: start")
    write_csv(dataset_dir / "rec.csv", ["fnsz", "bckg"])
    write_edf(dataset_dir / "rec.edf")
    out = dataset_dir / "master.csv"
    df = make_master_file(dataset_dir, output_path=str(out))
    assert df is not None
    assert out.exists()
    assert set(df["label"]) == {"fnsz", "bckg"}
    logger.info("test_make_master_file_basic: passed")

def test_make_master_file_skips_missing_edf(dataset_dir):
    logger.info("test_make_master_file_skips_missing_edf: start")
    write_csv(dataset_dir / "no_edf.csv", ["fnsz"])
    out = dataset_dir / "master.csv"
    df = make_master_file(dataset_dir, output_path=str(out))
    assert df is None
    logger.info("test_make_master_file_skips_missing_edf: passed")

def test_make_master_file_allow_tag_filters(dataset_dir):
    logger.info("test_make_master_file_allow_tag_filters: start")
    write_csv(dataset_dir / "rec.csv", ["fnsz", "bckg", "gnsz"])
    write_edf(dataset_dir / "rec.edf")
    out = dataset_dir / "master.csv"
    df = make_master_file(dataset_dir, output_path=str(out), allow_tag={"fnsz"})
    assert set(df["label"]) == {"fnsz"}
    logger.info("test_make_master_file_allow_tag_filters: passed")

def test_make_master_file_columns(dataset_dir):
    logger.info("test_make_master_file_columns: start")
    write_csv(dataset_dir / "rec.csv", ["fnsz"])
    write_edf(dataset_dir / "rec.edf")
    out = dataset_dir / "master.csv"
    df = make_master_file(dataset_dir, output_path=str(out))
    for col in ["edf_path", "csv_path", "split", "channel", "start_time", "stop_time", "label", "is_valid"]:
        assert col in df.columns, f"Missing column: {col}"
    logger.info("test_make_master_file_columns: passed")

def test_make_master_file_empty_dir(dataset_dir):
    logger.info("test_make_master_file_empty_dir: start")
    df = make_master_file(dataset_dir, output_path=str(dataset_dir / "master.csv"))
    assert df is None
    logger.info("test_make_master_file_empty_dir: passed")

def test_make_master_file_ictal_is_valid(dataset_dir):
    logger.info("test_make_master_file_ictal_is_valid: start")
    write_csv(dataset_dir / "rec.csv", ["fnsz"])
    write_edf(dataset_dir / "rec.edf")
    out = dataset_dir / "master.csv"
    df = make_master_file(dataset_dir, output_path=str(out))
    assert (df["is_valid"] == True).all()
    logger.info("test_make_master_file_ictal_is_valid: passed")

def test_make_master_file_split_assigned(dataset_dir):
    logger.info("test_make_master_file_split_assigned: start")
    train_dir = dataset_dir / "edf" / "train" / "01_tcp_ar"
    train_dir.mkdir(parents=True)
    write_csv(train_dir / "rec.csv", ["fnsz"])
    write_edf(train_dir / "rec.edf")
    out = dataset_dir / "master.csv"
    df = make_master_file(dataset_dir, output_path=str(out))
    assert (df["split"] == "train").all()
    logger.info("test_make_master_file_split_assigned: passed")


def test_chop_master_windows_fixed_length():
    from pipeline.preictal_segment import chop_master_windows

    master = pd.DataFrame([
        {"split": "train", "edf_path": "a.edf", "channel": "0", "start_time": 0.0, "stop_time": 10.0, "label": "pfnsz", "is_valid": True},
        {"split": "train", "edf_path": "a.edf", "channel": "0", "start_time": 20.0, "stop_time": 28.0, "label": "fnsz", "is_valid": True},
        {"split": "train", "edf_path": "a.edf", "channel": "0", "start_time": 30.0, "stop_time": 34.0, "label": "bg", "is_valid": True},
        {"split": "train", "edf_path": "a.edf", "channel": "0", "start_time": 100.0, "stop_time": 120.0, "label": "pfnsz_sopbuffer", "is_valid": False},
    ])

    chopped = chop_master_windows(master, window_duration=4.0)

    # valid rows should all be exactly 4.0s
    valid_chopped = chopped[chopped["is_valid"] & ~chopped["label"].str.endswith("_sopbuffer")]
    assert (valid_chopped["stop_time"] - valid_chopped["start_time"] == 4.0).all()

    # preictal 0..10 -> 2 windows (0..4, 4..8), 8..10 dropped
    # fnsz 20..28 -> 2 windows (20..24, 24..28)
    # bg 30..34 -> 1 window (30..34)
    # sopbuffer -> preserved
    assert len(chopped[chopped["label"] == "pfnsz"]) == 2
    assert len(chopped[chopped["label"] == "fnsz"]) == 2
    assert len(chopped[chopped["label"] == "bg"]) == 1
    assert len(chopped[chopped["label"] == "pfnsz_sopbuffer"]) == 1