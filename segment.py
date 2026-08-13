import numpy as np
import pandas as pd
from util import handle_logs

logger = handle_logs.get_logger("segment_npy", "applog")

TARGET_SFREQ = 256  # must match raw_eeg_extraction.TARGET_SFREQ


def _normalize_edf_path(edf_path):
    """
    Canonicalize an edf_path for matching only (never for I/O). The
    session index is always built with the current OS's separator
    (os.path.join), but a master CSV can be generated on a different
    OS than the one it's later consumed on (e.g. built on Windows,
    consumed on Linux) - a raw string-equality match would then silently
    fail for every row. Normalizing both sides to forward slashes before
    comparing makes matching OS-independent.
    """
    return str(edf_path).replace("\\", "/")


def build_edf_to_session_map(sessions):
    """sessions: dict[session_key] -> {"edf_paths": [...]}, from index_sessions().
    Keys are normalized (see _normalize_edf_path) for OS-independent matching."""
    mapping = {}
    for session_key, session in sessions.items():
        for edf_path in session["edf_paths"]:
            mapping[_normalize_edf_path(edf_path)] = session_key
    return mapping


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


def extract_segments(master_df, sessions, session_data, label_filter=None,
                     is_valid_filter=True, dedup_channels=True):
    """
    Slice labeled segments out of per-session concatenated in-memory arrays
    using the time annotations in master_df (e.g. master_full.csv).

    Parameters:
        master_df: rows already filtered/loaded from master_full.csv.
        sessions: dict from index_sessions(), used to map edf_path -> session_key.
        session_data: dict[session_key] -> (combined, file_offsets), i.e. the
            exact (np.ndarray, list[dict]) tuple returned by
            raw_eeg_extraction.concatenate_session_eeg() for that session.
            Nothing is read from disk here - the raw arrays are expected to
            still be held in memory from the extraction step.
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
        edf_path_norm = _normalize_edf_path(edf_path)

        session_key = edf_to_session.get(edf_path_norm)
        if session_key is None:
            logger.warning(f"No session found for {edf_path} - skipping {len(group)} segments")
            continue

        session_entry = session_data.get(session_key)
        if session_entry is None:
            logger.warning(f"No in-memory data for session {session_key} - skipping {len(group)} segments")
            continue

        combined, file_offsets = session_entry
        offsets_by_edf = {_normalize_edf_path(o["edf_path"]): o for o in file_offsets}

        offset = offsets_by_edf.get(edf_path_norm)
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