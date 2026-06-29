import pytest
import pandas as pd
from pipeline.preictal_segment import get_unique_tags, make_master_file, add_preictal_tags
from util import handle_logs

logger = handle_logs.get_logger("test_pipeline", "logs/test.log")

# Helpers 

def write_csv(path, labels, start_times=None, stop_times=None):
    n = len(labels)
    df = pd.DataFrame({
        "channel":    ["FP1-F7"] * n,
        "start_time": start_times if start_times else [0.0] * n,
        "stop_time":  stop_times  if stop_times  else [1.0] * n,
        "label":      labels,
        "confidence": [1] * n,
    })
    df.to_csv(path, index=False)

def write_edf(path):
    """Write a dummy .edf file so make_master_file doesn't skip the CSV."""
    path.write_text("dummy edf")


@pytest.fixture
def dataset_dir(tmp_path):
    return tmp_path

def test_logging_creates_file(tmp_path):
    logger.info("test_logging_creates_file: start")
    log_file = tmp_path / "test.log"
    test_logger = handle_logs.get_logger("test_logger", str(log_file))
    test_logger.info("This is a test log message.")
    assert log_file.exists(), "Log file was not created"
    logger.info("test_logging_creates_file: passed")

def test_logging_writes_content(tmp_path):
    logger.info("test_logging_writes_content: start")
    log_file = tmp_path / "test.log"
    test_logger = handle_logs.get_logger("test_logger_content", str(log_file))
    test_logger.info("hello from test")
    assert "hello from test" in log_file.read_text()
    logger.info("test_logging_writes_content: passed")

def test_single_file_single_tag(dataset_dir):
    logger.info("test_single_file_single_tag: start")
    write_csv(dataset_dir / "test.csv", ["bckg"])
    assert get_unique_tags(dataset_dir) == {"bckg"}
    logger.info("test_single_file_single_tag: passed")

def test_single_file_multiple_tags(dataset_dir):
    logger.info("test_single_file_multiple_tags: start")
    write_csv(dataset_dir / "test.csv", ["bckg", "fnsz", "gnsz"])
    assert get_unique_tags(dataset_dir) == {"bckg", "fnsz", "gnsz"}
    logger.info("test_single_file_multiple_tags: passed")

def test_multiple_files_deduped(dataset_dir):
    logger.info("test_multiple_files_deduped: start")
    write_csv(dataset_dir / "a.csv", ["bckg", "fnsz"])
    write_csv(dataset_dir / "b.csv", ["fnsz", "gnsz"])
    assert get_unique_tags(dataset_dir) == {"bckg", "fnsz", "gnsz"}
    logger.info("test_multiple_files_deduped: passed")

def test_nested_directories(dataset_dir):
    logger.info("test_nested_directories: start")
    subdir = dataset_dir / "subdir"
    subdir.mkdir()
    write_csv(dataset_dir / "a.csv", ["bckg"])
    write_csv(subdir / "b.csv", ["fnsz"])
    assert get_unique_tags(dataset_dir) == {"bckg", "fnsz"}
    logger.info("test_nested_directories: passed")

def test_empty_directory(dataset_dir):
    logger.info("test_empty_directory: start")
    assert get_unique_tags(dataset_dir) == set()
    logger.info("test_empty_directory: passed")

def test_ignores_non_csv_files(dataset_dir):
    logger.info("test_ignores_non_csv_files: start")
    (dataset_dir / "test.edf").write_text("not a csv")
    assert get_unique_tags(dataset_dir) == set()
    logger.info("test_ignores_non_csv_files: passed")

def test_malformed_csv_skipped(dataset_dir):
    logger.info("test_malformed_csv_skipped: start")
    (dataset_dir / "bad.csv").write_text("not,valid,csv\n!!!\n")
    write_csv(dataset_dir / "good.csv", ["bckg"])
    assert "bckg" in get_unique_tags(dataset_dir)
    logger.info("test_malformed_csv_skipped: passed")

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
    for col in ["edf_path", "csv_path", "channel", "start_time", "stop_time", "label"]:
        assert col in df.columns, f"Missing column: {col}"
    logger.info("test_make_master_file_columns: passed")

def test_make_master_file_empty_dir(dataset_dir):
    logger.info("test_make_master_file_empty_dir: start")
    df = make_master_file(dataset_dir, output_path=str(dataset_dir / "master.csv"))
    assert df is None
    logger.info("test_make_master_file_empty_dir: passed")

@pytest.fixture
def sample_master():
    return pd.DataFrame({
        "edf_path":   ["a.edf", "a.edf"],
        "csv_path":   ["a.csv", "a.csv"],
        "channel":    ["FP1-F7", "FP1-F7"],
        "start_time": [100.0, 400.0],
        "stop_time":  [110.0, 420.0],
        "label":      ["fnsz", "gnsz"],
        "confidence": [1, 1],
    })

def test_preictal_row_count(sample_master):
    logger.info("test_preictal_row_count: start")
    result = add_preictal_tags(sample_master, preictal_dur=300)
    assert len(result) == len(sample_master) * 2
    logger.info("test_preictal_row_count: passed")

def test_preictal_labels(sample_master):
    logger.info("test_preictal_labels: start")
    result = add_preictal_tags(sample_master, preictal_dur=300)
    assert "pfnsz" in result["label"].values
    assert "pgnsz" in result["label"].values
    logger.info("test_preictal_labels: passed")

def test_preictal_times(sample_master):
    logger.info("test_preictal_times: start")
    result = add_preictal_tags(sample_master, preictal_dur=300)
    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 0.0   # max(0, 100 - 300)
    assert pfnsz["stop_time"]  == 100.0
    pgnsz = result[result["label"] == "pgnsz"].iloc[0]
    assert pgnsz["start_time"] == 100.0  # max(0, 400 - 300)
    assert pgnsz["stop_time"]  == 400.0
    logger.info("test_preictal_times: passed")

def test_preictal_clipped_to_zero(sample_master):
    logger.info("test_preictal_clipped_to_zero: start")
    result = add_preictal_tags(sample_master, preictal_dur=9999)
    preictal_rows = result[result["label"].str.startswith("p")]
    assert (preictal_rows["start_time"] >= 0).all()
    logger.info("test_preictal_clipped_to_zero: passed")

def test_preictal_sorted_by_time(sample_master):
    logger.info("test_preictal_sorted_by_time: start")
    result = add_preictal_tags(sample_master, preictal_dur=300)
    times = result["start_time"].tolist()
    assert times == sorted(times)
    logger.info("test_preictal_sorted_by_time: passed")