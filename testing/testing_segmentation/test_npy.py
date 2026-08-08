import json

import numpy as np
import pandas as pd

import pipeline.segment_npy as seg  

def check_session_bookkeeping(session_key, output_dir):
    combined, offsets_by_edf = seg.load_session_data(session_key, output_dir)
    offsets = sorted(offsets_by_edf.values(), key=lambda o: o["start_sample"])
    issues = {}
    if offsets and offsets[-1]["end_sample"] != combined.shape[1]:
        issues["length_mismatch"] = True
    if any(o["end_sample"] <= o["start_sample"] for o in offsets):
        issues["zero_or_negative_width"] = True
    if any(b["start_sample"] < a["end_sample"] for a, b in zip(offsets, offsets[1:])):
        issues["overlapping_segments"] = True
    return issues


def check_boundary_discontinuities(session_key, output_dir, z_thresh=8.0):
    combined, offsets_by_edf = seg.load_session_data(session_key, output_dir)
    offsets = sorted(offsets_by_edf.values(), key=lambda o: o["start_sample"])
    baseline = np.median(np.abs(np.diff(combined, axis=1)).mean(axis=0)) + 1e-9
    return [
        b for a, b in zip(offsets, offsets[1:])
        if 0 < b["start_sample"] < combined.shape[1]
        and np.abs(combined[:, b["start_sample"]] - combined[:, b["start_sample"] - 1]).mean() / baseline > z_thresh
    ]


def check_segment_durations_against_csv(master_df, sessions, output_dir, tolerance_samples=2):
    results = seg.extract_segments(master_df, sessions, output_dir, dedup_channels=False)
    mismatches = []
    for r in results:
        expected = round((r["stop_time"] - r["start_time"]) * seg.TARGET_SFREQ)
        if abs(r["segment"].shape[1] - expected) > tolerance_samples:
            mismatches.append(r)
    return mismatches


# --- helpers ---

def write_session(tmp_path, key, combined, offsets):
    np.save(tmp_path / f"{key}_raw.npy", combined)
    json.dump(offsets, open(tmp_path / f"{key}_offsets.json", "w"))


def offset(edf, start, end):
    return {"edf_path": edf, "start_sample": start, "end_sample": end}


# --- tests ---

def test_bookkeeping_catches_length_overlap_and_zero_width(tmp_path):
    combined = np.random.randn(2, 900)  # shorter than offsets claim
    offsets = [offset("a.edf", 0, 500), offset("b.edf", 400, 400), offset("c.edf", 400, 1000)]
    write_session(tmp_path, "sess", combined, offsets)

    issues = check_session_bookkeeping("sess", tmp_path)

    assert "length_mismatch" in issues
    assert "overlapping_segments" in issues
    assert "zero_or_negative_width" in issues


def test_boundary_check_flags_jump_but_not_smooth_signal(tmp_path):
    offsets = [offset("a.edf", 0, 400), offset("b.edf", 400, 1000)]

    jumpy = np.zeros((2, 1000))
    jumpy[:, 400:] = 1000.0
    write_session(tmp_path, "sessJump", jumpy, offsets)
    assert len(check_boundary_discontinuities("sessJump", tmp_path)) == 1

    smooth = np.tile(np.sin(np.linspace(0, 10 * np.pi, 1000)), (2, 1))
    write_session(tmp_path, "sessSmooth", smooth, offsets)
    assert check_boundary_discontinuities("sessSmooth", tmp_path) == []


def test_csv_consistency_catches_clipped_segment(tmp_path):
    sfreq = seg.TARGET_SFREQ
    write_session(tmp_path, "sess1", np.random.randn(2, 20 * sfreq), [offset("a.edf", 0, 20 * sfreq)])
    sessions = {"sess1": {"edf_paths": ["a.edf"]}}

    df = pd.DataFrame([
        {"edf_path": "a.edf", "start_time": 2.0, "stop_time": 6.0,
         "label": "fnsz", "is_valid": True, "channel": "FP1-F7"},          # fits fine
        {"edf_path": "a.edf", "start_time": 18.0, "stop_time": 25.0,       # claims 7s, only 2s left
         "label": "fnsz", "is_valid": True, "channel": "FP1-F7"},
    ])

    mismatches = check_segment_durations_against_csv(df, sessions, tmp_path)

    assert len(mismatches) == 1
    assert mismatches[0]["start_time"] == 18.0