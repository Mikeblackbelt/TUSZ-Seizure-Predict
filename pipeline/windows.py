from util import handle_logs

logger = handle_logs.get_logger("segment_windows", "applog")


def _segment_one(arr, seg_len, step):
    """Slice a single (n_channels, n_samples) array into seg_len-length
    chunks, step samples apart. Drops any trailing samples that don't
    fill a full segment - no partial segments."""
    n_samples = arr.shape[1]
    if n_samples < seg_len:
        return []
    return [arr[:, start:start + seg_len]
            for start in range(0, n_samples - seg_len + 1, step)]


def segment_fixed(windows, seg_time, sfreq, step_time=None):
    """
    Non-adaptive segmentation: fixed-length segments at a fixed step.
    Use this for the class you're NOT trying to balance via overlap
    (e.g. interictal, the majority class) - default step_time=seg_time
    means no overlap at all, matching "13. split into 4 second segments"
    on its own.

    Parameters:
        windows: list of window dicts from extract_windows() (each with
            a "window" array plus metadata like label/status/edf_path).
        seg_time (float): segment length in seconds (SEG_TIME).
        sfreq (int): sample rate (must match the windows' actual sfreq).
        step_time (float or None): step between segment starts, in
            seconds. Defaults to seg_time (no overlap). Pass something
            smaller than seg_time for a fixed/manual overlap.

    Returns:
        list of dicts: original window metadata (minus "window") plus
        "segment" (the sliced array) and "seg_start_sample".
    """
    seg_len = int(seg_time * sfreq)
    step = int(step_time * sfreq) if step_time is not None else seg_len

    segments = []
    skipped = 0
    for w in windows:
        arr = w["window"]
        chunks = _segment_one(arr, seg_len, step)
        if not chunks:
            skipped += 1
            continue
        for i, chunk in enumerate(chunks):
            segments.append({
                **{k: v for k, v in w.items() if k != "window"},
                "segment": chunk,
                "seg_start_sample": i * step,
            })

    logger.info(
        f"segment_fixed: {len(windows)} windows -> {len(segments)} segments "
        f"(seg_len={seg_len}, step={step}), {skipped} windows too short to segment"
    )
    return segments


def compute_adaptive_step(total_len_pre, total_len_inter, seg_len):
    """
    Derive the overlap step for the minority (preictal) class so that
    overlapping segmentation produces roughly as many segments as the
    non-overlapping majority (interictal) class - balances class counts
    via overlap instead of duplication.

        step_for_overlap = int(SEG_LEN / (total_len_inter / total_len_pre))

    Parameters:
        total_len_pre (int): total samples across all preictal windows
            (sum of each window's arr.shape[1]).
        total_len_inter (int): total samples across all interictal
            windows, same units.
        seg_len (int): segment length in samples (SEG_LEN).

    Returns:
        int: step size in samples, clamped to >=1. If total_len_pre is
        so much larger than total_len_inter that the ratio would push
        step above seg_len (i.e. LESS overlap than non-adaptive), it's
        clamped to seg_len instead - adaptive overlap should never
        produce fewer segments than the non-overlapping case.
    """
    if total_len_pre <= 0:
        raise ValueError("total_len_pre must be > 0")
    ratio = total_len_inter / total_len_pre
    step = int(seg_len / ratio) if ratio > 0 else seg_len
    step = max(1, min(step, seg_len))
    logger.info(
        f"compute_adaptive_step: total_len_pre={total_len_pre}, "
        f"total_len_inter={total_len_inter}, ratio={ratio:.4f} -> step={step} "
        f"(seg_len={seg_len}, overlap={seg_len - step})"
    )
    return step


def segment_adaptive(windows, seg_time, sfreq, total_len_inter, total_len_pre=None):
    """
    Adaptive overlapping segmentation: step is computed from the class
    imbalance ratio (total_len_inter / total_len_pre) so the minority
    class gets more overlapping segments, balancing counts against the
    non-overlapping majority class. Use this for preictal.

    Parameters:
        windows: list of window dicts from extract_windows() - the
            minority-class windows to segment (e.g. preictal_windows).
        seg_time (float): segment length in seconds (SEG_TIME).
        sfreq (int): sample rate.
        total_len_inter (int): total samples across ALL interictal
            windows (the majority class you're balancing against) -
            compute this once from the full interictal set, not a
            subset, or the ratio (and resulting balance) will be off.
        total_len_pre (int or None): total samples across these windows.
            If None, computed as sum(w["window"].shape[1] for w in windows).

    Returns:
        list of dicts, same shape as segment_fixed()'s output.
    """
    seg_len = int(seg_time * sfreq)
    if total_len_pre is None:
        total_len_pre = sum(w["window"].shape[1] for w in windows)

    step = compute_adaptive_step(total_len_pre, total_len_inter, seg_len)

    segments = []
    skipped = 0
    for w in windows:
        arr = w["window"]
        chunks = _segment_one(arr, seg_len, step)
        if not chunks:
            skipped += 1
            continue
        for i, chunk in enumerate(chunks):
            segments.append({
                **{k: v for k, v in w.items() if k != "window"},
                "segment": chunk,
                "seg_start_sample": i * step,
            })

    logger.info(
        f"segment_adaptive: {len(windows)} windows -> {len(segments)} segments "
        f"(seg_len={seg_len}, step={step}, overlap={seg_len - step}), "
        f"{skipped} windows too short to segment"
    )
    return segments


def segment_df_fixed(df, seg_time, step_time=None, label_filter=None):
    """
    Non-adaptive interval segmentation for a pandas DataFrame: chops rows
    with [start_time, stop_time] into fixed-length `seg_time` rows at `step_time`
    increments.

    Parameters:
        df (pd.DataFrame): DataFrame containing start_time, stop_time, label.
        seg_time (float): Segment duration in seconds.
        step_time (float or None): Step duration in seconds. Defaults to seg_time (non-overlapping).
        label_filter (iterable or None): If provided, only rows whose label is in
            label_filter will be segmented; other rows pass through unchanged.

    Returns:
        pd.DataFrame: New DataFrame with segmented rows.
    """
    import pandas as pd

    if df.empty:
        return df.copy()

    step = step_time if step_time is not None else seg_time
    if seg_time <= 0 or step <= 0:
        raise ValueError(f"seg_time and step_time must be positive, got {seg_time}, {step}")

    label_set = set(label_filter) if label_filter is not None else None

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        if label_set is not None and row["label"] not in label_set:
            rows.append(row.to_dict())
            continue

        s_time = float(row["start_time"])
        e_time = float(row["stop_time"])
        if e_time - s_time < seg_time - 1e-9:
            skipped += 1
            continue

        t = s_time
        while t + seg_time <= e_time + 1e-9:
            new_row = dict(row)
            new_row["start_time"] = t
            new_row["stop_time"] = t + seg_time
            rows.append(new_row)
            t += step

    logger.info(
        f"segment_df_fixed: {len(df)} input rows -> {len(rows)} segmented rows "
        f"(seg_time={seg_time}, step={step}), {skipped} rows too short to segment"
    )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=df.columns)


def segment_df_adaptive(df, seg_time, total_len_bg, total_len_pre=None, preictal_label_prefix="p"):
    """
    Adaptive overlapping interval segmentation for preictal rows in a DataFrame.
    Step size is computed relative to total_len_bg / total_len_pre to balance
    minority (preictal) counts against majority (background) counts.

    Parameters:
        df (pd.DataFrame): DataFrame containing start_time, stop_time, label.
        seg_time (float): Segment duration in seconds.
        total_len_bg (float): Total background duration (seconds).
        total_len_pre (float or None): Total preictal duration (seconds). If None, calculated from df.
        preictal_label_prefix (str): Label prefix identifying preictal rows (default "p").

    Returns:
        pd.DataFrame: New DataFrame with adaptively segmented preictal rows and untouched non-preictal rows.
    """
    import pandas as pd

    if df.empty:
        return df.copy()

    is_pre = df["label"].astype(str).str.startswith(preictal_label_prefix) & ~df["label"].astype(str).str.endswith("_sopbuffer")
    if not is_pre.any():
        return df.copy()

    if total_len_pre is None or total_len_pre <= 0:
        total_len_pre = (df.loc[is_pre, "stop_time"] - df.loc[is_pre, "start_time"]).sum()

    if total_len_pre <= 0 or total_len_bg <= 0:
        step = seg_time
    else:
        ratio = total_len_bg / total_len_pre
        step = seg_time / ratio if ratio > 0 else seg_time
        step = max(0.1, min(step, seg_time))

    logger.info(f"segment_df_adaptive: calculated step={step:.4f}s for preictal rows (seg_time={seg_time}s)")

    pre_df = df[is_pre]
    other_df = df[~is_pre]

    segmented_pre = segment_df_fixed(pre_df, seg_time=seg_time, step_time=step)
    result = pd.concat([other_df, segmented_pre], ignore_index=True)
    if all(c in result.columns for c in ["split", "edf_path", "channel", "start_time"]):
        result = result.sort_values(["split", "edf_path", "channel", "start_time"]).reset_index(drop=True)
    return result