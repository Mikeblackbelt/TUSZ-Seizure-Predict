import os
import pytest
import mne
import numpy as np
import pandas as pd
import tempfile

from pipeline.process_signal import load_edf
from util import handle_logs

logger = handle_logs.get_logger("test_edf_load", "logs/test.log")

@pytest.fixture
def test_edf_path():
    """Create a temporary EDF file for testing."""
    sfreq = 256
    n_samples = 1024
    n_channels = 8
    ch_names = ['T3', 'C3', 'CZ', 'C4', 'T4','T5', 'P3', 'PZ']
    
    data = np.random.randn(n_channels, n_samples)
    info = mne.create_info(ch_names, sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(data, info)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
        raw.export(tmp.name, overwrite=True)
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Cleanup after test
    os.unlink(tmp_path)

def test_load_edf_metadata_columns(test_edf_path):
    logger.info("test_load_edf_metadata_columns: start")
    _, metadata = load_edf(test_edf_path)
    expected_columns = ['path', 'channels', 'sfreq', 'n_samples', 'duration_sec']
    for col in expected_columns:
        assert col in metadata.columns
    logger.info("test_load_edf_metadata_columns: passed")

def test_load_edf_returns(test_edf_path):
    logger.info("test_load_edf_returns: start")
    result = load_edf(test_edf_path)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], mne.io.BaseRaw)
    assert isinstance(result[1], pd.DataFrame)
    logger.info("test_load_edf_returns: passed")

def test_load_edf_metadata_has_one_row(test_edf_path):
    logger.info("test_load_edf_metadata_has_one_row: start")
    _, metadata = load_edf(test_edf_path)
    assert len(metadata) == 1
    logger.info("test_load_edf_metadata_has_one_row: passed")



