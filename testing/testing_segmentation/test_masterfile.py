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
        "status":     [-1, -1],
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
    for col in ["edf_path", "csv_path", "split", "channel", "start_time", "stop_time", "label", "status"]:
        assert col in df.columns, f"Missing column: {col}"
    logger.info("test_make_master_file_columns: passed")

def test_make_master_file_empty_dir(dataset_dir):
    logger.info("test_make_master_file_empty_dir: start")
    df = make_master_file(dataset_dir, output_path=str(dataset_dir / "master.csv"))
    assert df is None
    logger.info("test_make_master_file_empty_dir: passed")

def test_make_master_file_status_not_applicable(dataset_dir):
    logger.info("test_make_master_file_status_not_applicable: start")
    write_csv(dataset_dir / "rec.csv", ["fnsz"])
    write_edf(dataset_dir / "rec.edf")
    out = dataset_dir / "master.csv"
    df = make_master_file(dataset_dir, output_path=str(out))
    assert (df["status"] == -1).all()
    logger.info("test_make_master_file_status_not_applicable: passed")

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