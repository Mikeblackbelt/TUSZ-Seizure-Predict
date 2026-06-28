import pytest
from util import handle_logs
import pandas as pd
from pipeline import *

def test_logging(tmp_path):
    log_file = tmp_path / "test.log"

    logger = handle_logs.get_logger("test_logger", str(log_file))
    logger.info("This is a test log message.")

    assert log_file.exists()

@pytest.fixture
def dataset_dir(tmp_path):
    return tmp_path


def write_csv(path, labels):
    df = pd.DataFrame({
        "channel": ["FP1-F7"] * len(labels),
        "start_time": [0.0] * len(labels),
        "stop_time": [1.0] * len(labels),
        "label": labels,
        "confidence": [1] * len(labels),
    })
    df.to_csv(path, index=False)



+

def test_single_file_single_tag(dataset_dir):
    write_csv(dataset_dir / "test.csv", ["bckg"])
    assert get_unique_tags(dataset_dir) == {"bckg"}


def test_single_file_multiple_tags(dataset_dir):
    write_csv(dataset_dir / "test.csv", ["bckg", "fnsz", "gnsz"])
    assert get_unique_tags(dataset_dir) == {"bckg", "fnsz", "gnsz"}


def test_multiple_files_deduped(dataset_dir):
    write_csv(dataset_dir / "a.csv", ["bckg", "fnsz"])
    write_csv(dataset_dir / "b.csv", ["fnsz", "gnsz"])
    assert get_unique_tags(dataset_dir) == {"bckg", "fnsz", "gnsz"}


def test_nested_directories(dataset_dir):
    subdir = dataset_dir / "subdir"
    subdir.mkdir()
    write_csv(dataset_dir / "a.csv", ["bckg"])
    write_csv(subdir / "b.csv", ["fnsz"])
    assert get_unique_tags(dataset_dir) == {"bckg", "fnsz"}


def test_empty_directory(dataset_dir):
    assert get_unique_tags(dataset_dir) == set()


def test_ignores_non_csv_files(dataset_dir):
    (dataset_dir / "test.edf").write_text("not a csv")
    assert get_unique_tags(dataset_dir) == set()


def test_malformed_csv_skipped(dataset_dir):
    (dataset_dir / "bad.csv").write_text("not,valid,csv\n!!!\n")
    write_csv(dataset_dir / "good.csv", ["bckg"])
    assert "bckg" in get_unique_tags(dataset_dir)
    