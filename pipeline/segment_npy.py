import json
import numpy as np
import pandas as pd
from pathlib import Path
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
    Load the concatenated raw array and its per-file offsets for a session.

    ASSUMPTION: save_checkpoint() writes "{session_key}_raw.npy" and
    save_offsets() writes "{session_key}_offsets.json" containing the
    file_offsets list as-is. Adjust the two paths below if your actual
    util.handle_logs implementations name/serialize things differently.
    """
    combined = np.load(Path(output_dir) / f"{session_key}_raw.npy")
    with open(Path(output_dir) / f"{session_key}_offsets.json") as f:
        offsets = json.load(f)
    return combined, {o["edf_path"]: o for o in offsets}


def time_to_sample(t, offset, target_sfreq=TARGET_SFREQ):
    """
    Map a within-file time (seconds, relative to that edf's own start - the
    same convention make_master_file()/add_preictal_tags() use) to a sample
    index in the session's concatenated array. Clipped to the file's own
    [start_sample, end_sample) span so per-file resample rounding can never
    spill a segment into the neighboring file's samples.
    """
    sample = offset["start_sample"] + round(t * target_sfreq)
    return min(max(sample, offset["start_sample"]), offset["end_sample"])


def extract_segments(master_df, sessions, output_dir, label_filter=None,
                     is_valid_filter=True, dedup_channels=True):
    """
    Slice labeled segments out of per-session concatenated .npy arrays using
    the time annotations in master_df (e.g. master_full.csv).

    Parameters:
        master_df: rows already filtered/loaded from master_full.csv.
        sessions: dict from index_sessions(), used to map edf_path -> session_key.
        output_dir: dir containing "{session_key}_raw.npy" + offsets.
        label_filter: optional iterable of exact label values to keep.
        is_valid_filter: if True (default) only keep the valid segments 
        in ictal (all since it's already from the csv files), preictal, interictal  
        dedup_channels: if True (default), collapse rows that share the
            same (edf_path, start_time, stop_time, label) - i.e. the same
            segment annotated once per channel - into a single full-channel
            extraction. If False, slice once per row (still full-channel;
            `channel` is carried through as metadata either way since the
            array itself is never split by channel here).

    Returns:
        list of dicts: {"segment": np.ndarray (N_TARGET_CHANNELS, n_samples),
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

        offset = offsets_by_edf.get(edf_path)
        if offset is None:
            logger.warning(f"No file offset recorded for {edf_path} in session {session_key} - skipping")
            continue

        for _, row in group.iterrows():
            start_sample = time_to_sample(row["start_time"], offset)
            stop_sample = time_to_sample(row["stop_time"], offset)
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