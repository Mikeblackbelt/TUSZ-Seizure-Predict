import json
import numpy as np
import pytest

from pipeline import checkpoint_io
from util import handle_logs

logger = handle_logs.get_logger("test_checkpoint_io_offsets", "applog")
@pytest.fixture
def dataset_dir(tmp_path):
    return tmp_path

def test_save_and_load_offsets_round_trip(dataset_dir):
    file_offsets = [
        {"edf_path": "a.edf", "start_sample": 0, "end_sample": 100},
        {"edf_path": "b.edf", "start_sample": 100, "end_sample": 250},
    ]

    checkpoint_io.save_offsets(file_offsets, "sess001", str(dataset_dir))
    loaded = checkpoint_io.load_offsets("sess001", str(dataset_dir))
    assert loaded == file_offsets


def test_save_offsets_creates_output_dir(dataset_dir):

    nested_dir = dataset_dir / "nested" / "output"
    file_offsets = [{"edf_path": "a.edf", "start_sample": 0, "end_sample": 50}]
    out_path = checkpoint_io.save_offsets(file_offsets, "sess002", str(nested_dir))

    assert nested_dir.exists()
    with open(out_path) as f:
        assert json.load(f) == file_offsets


def test_load_offsets_missing_file_raises(dataset_dir):
    with pytest.raises(FileNotFoundError):
        checkpoint_io.load_offsets("nonexistent_session", str(dataset_dir))


def test_save_checkpoint_stores_compact_float32(tmp_path):
    array = np.arange(12, dtype=np.float64).reshape(3, 4)

    out_path = checkpoint_io.save_checkpoint(array, "sess003", str(tmp_path), stage="raw")

    with np.load(out_path) as data:
        saved = data["data"]
        assert saved.dtype == np.float32
        assert np.allclose(saved, array.astype(np.float32))