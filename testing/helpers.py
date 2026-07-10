import pytest
import pandas as pd
from pipeline.preictal_segment import (
    get_unique_tags,
    make_master_file,
    add_preictal_tags,
    get_split,
)
from util import handle_logs

logger = handle_logs.get_logger("test_pipeline", "app")

def write_csv(path, labels, start_times=None, stop_times=None):
    """
    Write a labeled CSV file with default timing and confidence values.
    
    Parameters:
    	path: File path where the CSV is written.
    	labels: Label values to store in the file.
    	start_times: Row start times to write, or default values of 0.0.
    	stop_times: Row stop times to write, or default values of 1.0.
    """
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
    """
    Create a dummy EDF file.
    
    Parameters:
    	path: File path to write.
    """
    path.write_text("dummy edf")


@pytest.fixture
def dataset_dir(tmp_path):
    """
    Provide a temporary dataset directory for tests.
    
    Returns:
    	tmp_path: A temporary directory path.
    """
    return tmp_path

