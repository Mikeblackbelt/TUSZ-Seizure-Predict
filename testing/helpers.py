import pytest
import pandas as pd
from pipeline.preictal_segment import (
    get_unique_tags,
    make_master_file,
    add_preictal_tags,
    get_split,
)
from util import handle_logs

logger = handle_logs.get_logger("test_pipeline", "logs/test.log")

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

