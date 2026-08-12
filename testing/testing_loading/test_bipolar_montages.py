import numpy as np
import pytest
from pipeline.eeg_channels import N_TARGET_CHANNELS
from pipeline.bipolar_montages import create_bipolar_montages, BIPOLAR_PAIRS, channel_index_dict
from pipeline.checkpoint_io import save_checkpoint, load_checkpoint
from util import handle_logs

logger = handle_logs.get_logger("test_bipolar_montages", "logs/test.log")

SESSION_KEY = "trn_p001_s001_ar1"


@pytest.fixture
def checkpoint_dir(tmp_path):
    return str(tmp_path)


def write_fake_raw_checkpoint(session_key, checkpoint_dir, n_samples=100, seed=0):
    rng = np.random.default_rng(seed)
    array = rng.random((N_TARGET_CHANNELS, n_samples))
    save_checkpoint(array, session_key, checkpoint_dir, stage="raw")
    return array


def test_create_bipolar_montages_all_pairs_correct(checkpoint_dir):
    logger.info("test_create_bipolar_montages_all_pairs_correct: start")

    original = write_fake_raw_checkpoint(SESSION_KEY, checkpoint_dir)

    result = create_bipolar_montages(SESSION_KEY, checkpoint_dir)

    for i, (ch1, ch2) in enumerate(BIPOLAR_PAIRS):
        idx1 = channel_index_dict[ch1]
        idx2 = channel_index_dict[ch2]
        expected = original[idx1, :] - original[idx2, :]
        assert np.allclose(result[i, :], expected), f"Row {i} ({ch1}-{ch2}) incorrect"

    logger.info("test_create_bipolar_montages_all_pairs_correct: passed")


def test_create_bipolar_montages_returns_none_on_missing_file(checkpoint_dir):
    logger.info("test_create_bipolar_montages_returns_none_on_missing_file: start")
    result = create_bipolar_montages("does_not_exist_session", checkpoint_dir)
    assert result is None
    logger.info("test_create_bipolar_montages_returns_none_on_missing_file: passed")


def test_create_bipolar_montages_returns_none_on_wrong_channel_count(checkpoint_dir):
    logger.info("test_create_bipolar_montages_returns_none_on_wrong_channel_count: start")
    bad_array = np.random.rand(N_TARGET_CHANNELS - 1, 100)
    save_checkpoint(bad_array, SESSION_KEY, checkpoint_dir, stage="raw")

    result = create_bipolar_montages(SESSION_KEY, checkpoint_dir)
    assert result is None
    logger.info("test_create_bipolar_montages_returns_none_on_wrong_channel_count: passed")


def test_create_bipolar_montages_saves_to_output_path(checkpoint_dir):
    logger.info("test_create_bipolar_montages_saves_to_output_path: start")
    write_fake_raw_checkpoint(SESSION_KEY, checkpoint_dir)

    result = create_bipolar_montages(SESSION_KEY, checkpoint_dir)

    saved = load_checkpoint(SESSION_KEY, checkpoint_dir, stage="proc")
    assert np.allclose(saved, result)
    logger.info("test_create_bipolar_montages_saves_to_output_path: passed")
