"""Unit tests for pipeline.artifact_detection.ArtifactDetector.

ArtifactDetector.__init__ requires a trained checkpoint (best_model.pth,
scaler.pkl) on disk, which isn't available in this environment/CI. These
tests bypass __init__ (via object.__new__) and inject lightweight stand-ins
for the model/scaler so the surrounding resampling, windowing, and
aggregation logic can be exercised in isolation.
"""
import numpy as np
import pytest
import torch

from pipeline.artifact_detection import ArtifactDetector
from util import handle_logs

logger = handle_logs.get_logger("test_artifact_detection", "logs/test.log")


class _IdentityScaler:
    """Stand-in for a fitted sklearn StandardScaler."""

    def transform(self, X):
        return X


class _ConstantLogitModel(torch.nn.Module):
    """Stand-in model that ignores its input and returns fixed logits for
    every window, regardless of feature size."""

    def __init__(self, logits=(5.0, -5.0, -5.0)):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits.unsqueeze(0).repeat(x.shape[0], 1)


@pytest.fixture
def detector():
    """A bare ArtifactDetector with stubbed model/scaler (no checkpoint I/O)."""
    det = ArtifactDetector.__new__(ArtifactDetector)
    det.device = torch.device("cpu")
    det.scaler = _IdentityScaler()
    det.use_pca = False
    det.pca = None
    det.model = _ConstantLogitModel()
    det.model.eval()
    return det


def test_resample_to_256_is_noop_when_already_256(detector):
    logger.info("test_resample_to_256_is_noop_when_already_256: start")
    channel = np.arange(100, dtype=np.float64)
    out = detector._resample_to_256(channel, 256)
    np.testing.assert_array_equal(out, channel)
    logger.info("test_resample_to_256_is_noop_when_already_256: passed")


def test_resample_to_256_changes_length_for_other_rates(detector):
    logger.info("test_resample_to_256_changes_length_for_other_rates: start")
    channel = np.sin(2 * np.pi * 10 * np.arange(512) / 512.0)
    out = detector._resample_to_256(channel, 512)
    assert len(out) == pytest.approx(256, abs=1)
    logger.info("test_resample_to_256_changes_length_for_other_rates: passed")


def test_segment_into_windows_shape(detector):
    logger.info("test_segment_into_windows_shape: start")
    sig = np.arange(512 * 3, dtype=np.float64)
    windows = detector._segment_into_windows(sig)
    assert windows.shape == (3, 512)
    logger.info("test_segment_into_windows_shape: passed")


def test_segment_into_windows_drops_incomplete_remainder(detector):
    logger.info("test_segment_into_windows_drops_incomplete_remainder: start")
    sig = np.arange(512 * 2 + 100, dtype=np.float64)
    windows = detector._segment_into_windows(sig)
    assert windows.shape == (2, 512)
    np.testing.assert_array_equal(windows[0], sig[:512])
    np.testing.assert_array_equal(windows[1], sig[512:1024])
    logger.info("test_segment_into_windows_drops_incomplete_remainder: passed")


def test_segment_into_windows_empty_when_shorter_than_one_window(detector):
    logger.info("test_segment_into_windows_empty_when_shorter_than_one_window: start")
    sig = np.arange(100, dtype=np.float64)
    windows = detector._segment_into_windows(sig)
    assert windows.shape == (0, 512)
    logger.info("test_segment_into_windows_empty_when_shorter_than_one_window: passed")


def test_predict_channel_returns_probabilities_summing_to_one(detector):
    logger.info("test_predict_channel_returns_probabilities_summing_to_one: start")
    rng = np.random.default_rng(0)
    channel = rng.standard_normal(512 * 4)

    probs = detector.predict_channel(channel, fs_in=256)

    assert probs.shape == (4, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-5)
    # Stub model heavily favors class 0.
    assert (probs.argmax(axis=1) == 0).all()
    logger.info("test_predict_channel_returns_probabilities_summing_to_one: passed")


def test_predict_channel_empty_when_signal_too_short(detector):
    logger.info("test_predict_channel_empty_when_signal_too_short: start")
    channel = np.zeros(100)
    probs = detector.predict_channel(channel, fs_in=256)
    assert probs.shape == (0, 3)
    logger.info("test_predict_channel_empty_when_signal_too_short: passed")


def test_predict_channel_resamples_non_native_rate(detector):
    logger.info("test_predict_channel_resamples_non_native_rate: start")
    rng = np.random.default_rng(1)
    # 4 windows worth of samples at 512 Hz -> resampled to 256 Hz.
    channel = rng.standard_normal(512 * 2 * 4)
    probs = detector.predict_channel(channel, fs_in=512)
    assert probs.shape == (4, 3)
    logger.info("test_predict_channel_resamples_non_native_rate: passed")


def test_predict_segment_handles_1d_input(detector):
    logger.info("test_predict_segment_handles_1d_input: start")
    rng = np.random.default_rng(2)
    eeg_data = rng.standard_normal(512 * 3)

    result = detector.predict_segment(eeg_data, fs_in=256)

    assert len(result["per_channel_probs"]) == 1
    assert result["per_channel_probs"][0].shape == (3, 3)
    assert result["total_windows"] == 3
    logger.info("test_predict_segment_handles_1d_input: passed")


def test_predict_segment_stacks_multiple_channels(detector):
    logger.info("test_predict_segment_stacks_multiple_channels: start")
    rng = np.random.default_rng(3)
    eeg_data = rng.standard_normal((2, 512 * 3))

    result = detector.predict_segment(eeg_data, fs_in=256)

    assert len(result["per_channel_probs"]) == 2
    assert result["total_windows"] == 6
    assert 0.0 <= result["artifact_fraction"] <= 1.0
    logger.info("test_predict_segment_stacks_multiple_channels: passed")


def test_predict_segment_artifact_fraction_is_low_for_clean_stub_predictions(detector):
    logger.info("test_predict_segment_artifact_fraction_is_low_for_clean_stub_predictions: start")
    rng = np.random.default_rng(4)
    eeg_data = rng.standard_normal((1, 512 * 2))

    result = detector.predict_segment(eeg_data, fs_in=256)

    # The stub model overwhelmingly predicts class 0 ("clean"), so the
    # artifact fraction (sum of prob mass on classes 1 and 2) should be near 0.
    assert result["artifact_fraction"] < 0.01
    logger.info("test_predict_segment_artifact_fraction_is_low_for_clean_stub_predictions: passed")


def test_predict_segment_no_windows_yields_zero_fraction(detector):
    logger.info("test_predict_segment_no_windows_yields_zero_fraction: start")
    eeg_data = np.zeros((2, 100))  # too short for even one window

    result = detector.predict_segment(eeg_data, fs_in=256)

    assert result["total_windows"] == 0
    assert result["artifact_fraction"] == 0.0
    logger.info("test_predict_segment_no_windows_yields_zero_fraction: passed")