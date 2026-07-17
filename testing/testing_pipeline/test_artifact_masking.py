"""Unit tests for pipeline.artifact_masking:
    build_artifact_mask, apply_zero_masking, apply_interpolation_masking
"""
import numpy as np
import pytest

from pipeline.artifact_masking import (
    WINDOW_SECONDS,
    build_artifact_mask,
    apply_zero_masking,
    apply_interpolation_masking,
)
from util import handle_logs

logger = handle_logs.get_logger("test_artifact_masking", "logs/test.log")


def make_probs(class_sequence, n_classes=3):
    """Build a fake (n_windows, n_classes) probability array whose argmax per
    row follows `class_sequence`."""
    probs = np.zeros((len(class_sequence), n_classes))
    for i, cls in enumerate(class_sequence):
        probs[i, cls] = 1.0
    return probs


def test_build_artifact_mask_shape_and_dtype():
    logger.info("test_build_artifact_mask_shape_and_dtype: start")
    fs_native = 256
    n_samples = 4 * int(round(WINDOW_SECONDS * fs_native))
    per_channel_probs = [make_probs([0, 0, 0, 0])]

    mask = build_artifact_mask(per_channel_probs, n_channels=1, n_samples_native=n_samples, fs_native=fs_native)

    assert mask.shape == (1, n_samples)
    assert mask.dtype == bool
    logger.info("test_build_artifact_mask_shape_and_dtype: passed")


def test_build_artifact_mask_flags_correct_windows():
    logger.info("test_build_artifact_mask_flags_correct_windows: start")
    fs_native = 256
    win_len = int(round(WINDOW_SECONDS * fs_native))
    n_windows = 3
    n_samples = n_windows * win_len
    # window 0 clean, window 1 artifact (class 1), window 2 clean
    per_channel_probs = [make_probs([0, 1, 0])]

    mask = build_artifact_mask(per_channel_probs, n_channels=1, n_samples_native=n_samples, fs_native=fs_native)

    assert not mask[0, :win_len].any()
    assert mask[0, win_len:2 * win_len].all()
    assert not mask[0, 2 * win_len:].any()
    logger.info("test_build_artifact_mask_flags_correct_windows: passed")


def test_build_artifact_mask_respects_artifact_classes_argument():
    logger.info("test_build_artifact_mask_respects_artifact_classes_argument: start")
    fs_native = 256
    win_len = int(round(WINDOW_SECONDS * fs_native))
    n_samples = 2 * win_len
    per_channel_probs = [make_probs([1, 2])]

    # class 1 not treated as an artifact here, only class 2
    mask = build_artifact_mask(
        per_channel_probs, n_channels=1, n_samples_native=n_samples, fs_native=fs_native,
        artifact_classes=(2,),
    )

    assert not mask[0, :win_len].any()
    assert mask[0, win_len:].all()
    logger.info("test_build_artifact_mask_respects_artifact_classes_argument: passed")


def test_build_artifact_mask_handles_empty_channel_probs():
    logger.info("test_build_artifact_mask_handles_empty_channel_probs: start")
    fs_native = 256
    n_samples = 512
    per_channel_probs = [np.zeros((0, 3))]

    mask = build_artifact_mask(per_channel_probs, n_channels=1, n_samples_native=n_samples, fs_native=fs_native)

    assert not mask.any()
    logger.info("test_build_artifact_mask_handles_empty_channel_probs: passed")


def test_build_artifact_mask_truncates_windows_beyond_n_samples():
    logger.info("test_build_artifact_mask_truncates_windows_beyond_n_samples: start")
    fs_native = 256
    win_len = int(round(WINDOW_SECONDS * fs_native))
    # Only give room for 1.5 windows worth of samples.
    n_samples = win_len + win_len // 2
    per_channel_probs = [make_probs([1, 1])]

    mask = build_artifact_mask(per_channel_probs, n_channels=1, n_samples_native=n_samples, fs_native=fs_native)

    assert mask.shape == (1, n_samples)
    assert mask.all()  # every available sample belongs to an artifact window
    logger.info("test_build_artifact_mask_truncates_windows_beyond_n_samples: passed")


def test_apply_zero_masking_zeros_flagged_and_preserves_clean_samples():
    logger.info("test_apply_zero_masking_zeros_flagged_and_preserves_clean_samples: start")
    fs_native = 256
    win_len = int(round(WINDOW_SECONDS * fs_native))
    n_samples = 2 * win_len
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1, n_samples))
    detector_result = {"per_channel_probs": [make_probs([0, 1])]}

    masked, mask = apply_zero_masking(data, detector_result, fs_native)

    assert masked.shape == data.shape
    assert (masked[mask] == 0.0).all()
    np.testing.assert_array_equal(masked[~mask], data[~mask])
    logger.info("test_apply_zero_masking_zeros_flagged_and_preserves_clean_samples: passed")


def test_apply_zero_masking_does_not_mutate_input():
    logger.info("test_apply_zero_masking_does_not_mutate_input: start")
    fs_native = 256
    win_len = int(round(WINDOW_SECONDS * fs_native))
    n_samples = win_len
    data = np.ones((1, n_samples))
    detector_result = {"per_channel_probs": [make_probs([1])]}

    masked, mask = apply_zero_masking(data, detector_result, fs_native)

    assert mask.all()
    assert (data == 1.0).all(), "original data array must not be modified in-place"
    assert (masked == 0.0).all()
    logger.info("test_apply_zero_masking_does_not_mutate_input: passed")


def test_apply_interpolation_masking_replaces_flagged_samples_and_preserves_others():
    logger.info("test_apply_interpolation_masking_replaces_flagged_samples_and_preserves_others: start")
    fs_native = 256
    win_len = int(round(WINDOW_SECONDS * fs_native))
    n_samples = 3 * win_len
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, n_samples))
    # channel 0: middle window flagged; channel 1: fully clean
    detector_result = {"per_channel_probs": [make_probs([0, 1, 0]), make_probs([0, 0, 0])]}

    interpolated, mask, fully_flagged = apply_interpolation_masking(data, detector_result, fs_native)

    assert interpolated.shape == data.shape
    assert fully_flagged == []
    np.testing.assert_array_equal(interpolated[~mask], data[~mask])
    assert not np.allclose(interpolated[mask], data[mask])
    logger.info("test_apply_interpolation_masking_replaces_flagged_samples_and_preserves_others: passed")


def test_apply_interpolation_masking_interpolates_linearly_between_clean_neighbors():
    logger.info("test_apply_interpolation_masking_interpolates_linearly_between_clean_neighbors: start")
    fs_native = 256
    win_len = int(round(WINDOW_SECONDS * fs_native))
    n_samples = 3 * win_len
    # A ramp signal so linear interpolation across the flagged window is
    # exactly predictable.
    data = np.tile(np.arange(n_samples, dtype=np.float64), (1, 1))
    detector_result = {"per_channel_probs": [make_probs([0, 1, 0])]}

    interpolated, mask, fully_flagged = apply_interpolation_masking(data, detector_result, fs_native)

    expected = np.interp(np.arange(n_samples), np.arange(n_samples)[~mask[0]], data[0, ~mask[0]])
    np.testing.assert_allclose(interpolated[0], expected)
    logger.info("test_apply_interpolation_masking_interpolates_linearly_between_clean_neighbors: passed")


def test_apply_interpolation_masking_fully_flagged_channel_left_unchanged():
    logger.info("test_apply_interpolation_masking_fully_flagged_channel_left_unchanged: start")
    fs_native = 256
    win_len = int(round(WINDOW_SECONDS * fs_native))
    n_samples = win_len
    rng = np.random.default_rng(2)
    data = rng.standard_normal((1, n_samples))
    detector_result = {"per_channel_probs": [make_probs([1])]}  # entire channel flagged

    interpolated, mask, fully_flagged = apply_interpolation_masking(data, detector_result, fs_native)

    assert fully_flagged == [0]
    np.testing.assert_array_equal(interpolated[0], data[0])
    logger.info("test_apply_interpolation_masking_fully_flagged_channel_left_unchanged: passed")


def test_zero_and_interpolation_masking_agree_on_which_samples_are_flagged():
    logger.info("test_zero_and_interpolation_masking_agree_on_which_samples_are_flagged: start")
    fs_native = 256
    win_len = int(round(WINDOW_SECONDS * fs_native))
    n_samples = 4 * win_len
    rng = np.random.default_rng(3)
    data = rng.standard_normal((2, n_samples))
    detector_result = {"per_channel_probs": [make_probs([0, 1, 2, 0]), make_probs([1, 0, 0, 2])]}

    _, mask_zero = apply_zero_masking(data, detector_result, fs_native)
    _, mask_interp, _ = apply_interpolation_masking(data, detector_result, fs_native)

    np.testing.assert_array_equal(mask_zero, mask_interp)
    logger.info("test_zero_and_interpolation_masking_agree_on_which_samples_are_flagged: passed")