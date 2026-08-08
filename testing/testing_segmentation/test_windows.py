import numpy as np
import pytest

from pipeline.windows import (
    _segment_one,
    segment_fixed,
    compute_adaptive_step,
    segment_adaptive,
)


def make_window(n_channels, n_samples, label="fnsz"):
    arr = np.random.randn(n_channels, n_samples).astype(np.float32)
    return {
        "window": arr,
        "label": label,
        "status": 1,
        "edf_path": "a.edf",
        "start_time": 0.0,
        "stop_time": n_samples / 256,
    }


def test_segment_one_exact_fit_no_overlap():
    arr = np.arange(20).reshape(1, 20)
    chunks = _segment_one(arr, seg_len=5, step=5)
    assert len(chunks) == 4
    assert all(c.shape == (1, 5) for c in chunks)


def test_segment_one_drops_trailing_remainder():
    arr = np.zeros((1, 22))
    chunks = _segment_one(arr, seg_len=5, step=5)
    assert len(chunks) == 4


def test_segment_one_too_short_returns_empty():
    arr = np.zeros((1, 3))
    chunks = _segment_one(arr, seg_len=5, step=5)
    assert chunks == []


def test_segment_one_chunk_content_matches_source():
    arr = np.arange(10).reshape(1, 10)
    chunks = _segment_one(arr, seg_len=4, step=4)
    assert np.array_equal(chunks[0], arr[:, 0:4])
    assert np.array_equal(chunks[1], arr[:, 4:8])


def test_segment_fixed_metadata_preserved_minus_window():
    windows = [make_window(2, 1024, label="pfnsz")]
    segs = segment_fixed(windows, seg_time=4.0, sfreq=256)
    assert segs[0]["label"] == "pfnsz"
    assert segs[0]["edf_path"] == "a.edf"
    assert "window" not in segs[0]
    assert "segment" in segs[0]


def test_segment_fixed_too_short_window_dropped_entirely():
    short = make_window(1, 100)
    normal = make_window(1, 1024)
    segs = segment_fixed([short, normal], seg_time=4.0, sfreq=256)
    assert len(segs) == 1


def test_compute_adaptive_step_equal_totals_gives_seg_len():
    step = compute_adaptive_step(total_len_pre=1000, total_len_inter=1000, seg_len=256)
    assert step == 256


def test_compute_adaptive_step_smaller_pre_class_gets_more_overlap():
    step = compute_adaptive_step(total_len_pre=1000, total_len_inter=5000, seg_len=256)
    assert step < 256


def test_compute_adaptive_step_clamped_to_seg_len_when_pre_larger():
    step = compute_adaptive_step(total_len_pre=5000, total_len_inter=1000, seg_len=256)
    assert step == 256


def test_compute_adaptive_step_raises_on_zero_pre():
    with pytest.raises(ValueError):
        compute_adaptive_step(total_len_pre=0, total_len_inter=1000, seg_len=256)


def test_segment_adaptive_balances_against_interictal_count():
    interictal_windows = [make_window(1, 4096) for _ in range(4)]
    total_len_inter = sum(w["window"].shape[1] for w in interictal_windows)
    interictal_segs = segment_fixed(interictal_windows, seg_time=4.0, sfreq=256)

    preictal_windows = [make_window(1, 4096)]
    preictal_segs = segment_adaptive(
        preictal_windows, seg_time=4.0, sfreq=256, total_len_inter=total_len_inter
    )
    preictal_segs_no_overlap = segment_fixed(preictal_windows, seg_time=4.0, sfreq=256)

    assert len(preictal_segs) > len(preictal_segs_no_overlap)
    assert len(preictal_segs) <= len(interictal_segs) * 2