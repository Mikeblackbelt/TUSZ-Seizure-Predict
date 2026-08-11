import numpy as np
import pytest
from pipeline.eeg_channels import N_TARGET_CHANNELS
from pipeline.bipolar_montages import create_bipolar_montages, BIPOLAR_PAIRS, channel_index_dict
from util import handle_logs

logger = handle_logs.get_logger("test_bipolar_montages", "logs/test.log")


@pytest.fixture
def checkpoint_dir(tmp_path):
    return tmp_path


def write_fake_raw_checkpoint(checkpoint_dir, session_key, n_samples=100, seed=0):
    rng = np.random.default_rng(seed)
    array = rng.random((N_TARGET_CHANNELS, n_samples))
    np.save(checkpoint_dir / f"{session_key}_raw.npy", array)
    return array


def test_create_bipolar_montages_all_pairs_correct(checkpoint_dir):
    logger.info("test_create_bipolar_montages_all_pairs_correct: start")

    session_key = "session"
    original = write_fake_raw_checkpoint(checkpoint_dir, session_key)

    result = create_bipolar_montages(session_key, str(checkpoint_dir))

    for i, (ch1, ch2) in enumerate(BIPOLAR_PAIRS):
        idx1 = channel_index_dict[ch1]
        idx2 = channel_index_dict[ch2]
        expected = original[idx1, :] - original[idx2, :]
        assert np.allclose(result[i, :], expected), f"Row {i} ({ch1}-{ch2}) incorrect"

    logger.info("test_create_bipolar_montages_all_pairs_correct: passed")


def test_create_bipolar_montages_returns_none_on_missing_file(checkpoint_dir):
    logger.info("test_create_bipolar_montages_returns_none_on_missing_file: start")
    result = create_bipolar_montages("does_not_exist", str(checkpoint_dir))
    assert result is None
    logger.info("test_create_bipolar_montages_returns_none_on_missing_file: passed")


def test_create_bipolar_montages_returns_none_on_wrong_channel_count(checkpoint_dir):
    logger.info("test_create_bipolar_montages_returns_none_on_wrong_channel_count: start")
    session_key = "bad_session"
    np.save(checkpoint_dir / f"{session_key}_raw.npy", np.random.rand(N_TARGET_CHANNELS - 1, 100))
    result = create_bipolar_montages(session_key, str(checkpoint_dir))
    assert result is None
    logger.info("test_create_bipolar_montages_returns_none_on_wrong_channel_count: passed")


def test_create_bipolar_montages_saves_proc_checkpoint(checkpoint_dir):
    logger.info("test_create_bipolar_montages_saves_proc_checkpoint: start")
    session_key = "session"
    write_fake_raw_checkpoint(checkpoint_dir, session_key)

    result = create_bipolar_montages(session_key, str(checkpoint_dir))

    proc_path = checkpoint_dir / f"{session_key}_proc.npy"
    assert proc_path.exists()
    assert np.allclose(np.load(proc_path), result)
    logger.info("test_create_bipolar_montages_saves_proc_checkpoint: passed")