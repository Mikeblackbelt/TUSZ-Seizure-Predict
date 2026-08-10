import json
import numpy as np
import pandas as pd
from pathlib import Path
from pipeline.checkpoint_io import load_checkpoint
from util import handle_logs

logger = handle_logs.get_logger("segment_npy", "applog")

TARGET_SFREQ = 256  # must match raw_eeg_extraction.TARGET_SFREQ


def build_edf_to_session_map(sessions):
    """sessions: dict[session_key] -> {"edf_paths": [...]}, from index_sessions()."""
    mapping = {}
    for session_key, session in sessions.items():
        for edf_path in session["edf_paths"]:
            mapping[edf_path] = session_key
    return mapping


def load_session_data(session_key, output_dir):
    """
    Load the bipolar-converted ('proc') checkpoint and its per-file offsets
    for a session.

    NOTE: this now reads the *proc* checkpoint (post bipolar_montages.py),
    not raw. Segmentation/windowing must happen on the bipolar-converted
    signal per the pipeline order (filter -> resample/concat -> bipolar ->
    exclusion -> window), so slicing from the 17-channel referential raw
    array here was a bug - bipolar_montages.py's output was never being
    consumed downstream.
    """
    combined = load_checkpoint(session_key, output_dir, stage="proc")
    with open(Path(output_dir) / f"{session_key}_offsets.json") as f:
        offsets = json.load(f)
    return combined, {o["edf_path"]: o for o in offsets}


def time_to_sample(t, edf_path, offsets_by_edf, target_sfreq=TARGET_SFREQ):
    """
    Map a within-file time (seconds, relative to that edf's own start - the
    same convention make_master_file()/add_preictal_tags() use) to a
    session-global sample index in the session's concatenated array.

    Unlike the original version, this is clipped to the *whole session's*
    [0, total_samples) span, not the single file's own [start_sample,
    end_sample) span. That's what allows preictal windows anchored near a
    file boundary to pull samples from the adjacent file: since the
    combined array is just files concatenated back-to-back, a negative t
    (before this file's start) or t beyond this file's own duration lands
    on real adjacent-file samples rather than getting clamped away.

    CAVEAT: this assumes recordings within a session are wall-clock
    contiguous (no real time gap between files, e.g. electrode
    reapplication, break in recording). The array is sample-contiguous by
    construction regardless, but if there IS a real gap, a cross-file
    preictal window will silently pull samples from the wrong wall-clock
    time. Worth checking recording start timestamps (if available in
    session_metadata / recording_info) before trusting cross-file
    preictal segments in training.
    """
    offset = offsets_by_edf[edf_path]
    sample = offset["start_sample"] + round(t * target_sfreq)
    total_samples = max(o["end_sample"] for o in offsets_by_edf.values())
    return min(max(sample, 0), total_samples)


def extract_segments(master_df, sessions, output_dir, label_filter=None,
                     is_valid_filter=True, dedup_channels=True):
    """
    Slice labeled segments out of per-session concatenated .npy arrays using
    the time annotations in master_df (e.g. master_full.csv).

    Parameters:
        master_df: rows already filtered/loaded from master_full.csv.
        sessions: dict from index_sessions(), used to map edf_path -> session_key.
        output_dir: dir containing "{session_key}_proc.npy" + offsets.
        label_filter: optional iterable of exact label values to keep.
            IMPORTANT: is_valid_filter alone does NOT exclude raw ictal
            rows - they're legitimately is_valid=True (see
            preictal_segment.resolve_overlaps()'s priority scheme; ictal
            outranks preictal/interictal and is only carved when
            overlapped, not dropped wholesale). To build a
            preictal/interictal-only training set, call this with
            label_filter=preictal_segment.get_trainable_labels(master_df).
        is_valid_filter: if True (default) only keep the valid segments.
        dedup_channels: if True (default), collapse rows that share the
            same (edf_path, start_time, stop_time, label) into a single
            full-channel extraction.

    Returns:
        list of dicts: {"segment": np.ndarray (N_BIPOLAR_CHANNELS, n_samples),
                         "label": str, "is_valid": bool, "edf_path": str,
                         "start_time": float, "stop_time": float,
                         "channels": list[str]}
    """
    df = master_df
    if label_filter is not None:
        df = df[df["label"].isin(label_filter)]
    if is_valid_filter is not None:
        df = df[df["is_valid"] == is_valid_filter]

    if dedup_channels:
        grouped = (
            df.groupby(["edf_path", "start_time", "stop_time", "label", "is_valid"])["channel"]
            .apply(list)
            .reset_index()
            .rename(columns={"channel": "channels"})
        )
    else:
        grouped = df.copy()
        grouped["channels"] = grouped["channel"].apply(lambda c: [c])

    edf_to_session = build_edf_to_session_map(sessions)

    results = []
    for edf_path, group in grouped.groupby("edf_path"):
        session_key = edf_to_session.get(edf_path)
        if session_key is None:
            logger.warning(f"No session found for {edf_path} - skipping {len(group)} segments")
            continue

        try:
            combined, offsets_by_edf = load_session_data(session_key, output_dir)
        except FileNotFoundError as e:
            logger.warning(f"Missing session data for {session_key}: {e} - skipping {len(group)} segments")
            continue

        if edf_path not in offsets_by_edf:
            logger.warning(f"No file offset recorded for {edf_path} in session {session_key} - skipping")
            continue

        for _, row in group.iterrows():
            start_sample = time_to_sample(row["start_time"], edf_path, offsets_by_edf)
            stop_sample = time_to_sample(row["stop_time"], edf_path, offsets_by_edf)
            if stop_sample <= start_sample:
                logger.warning(f"Degenerate segments for {edf_path} {row['label']} - skipping")
                continue

            results.append({
                "segment": combined[:, start_sample:stop_sample],
                "label": row["label"],
                "is_valid": row["is_valid"],
                "edf_path": edf_path,
                "start_time": row["start_time"],
                "stop_time": row["stop_time"],
                "channels": row["channels"],
            })

    logger.info(f"Extracted {len(results)} segments ({'deduped' if dedup_channels else 'per-row'})")
    return results