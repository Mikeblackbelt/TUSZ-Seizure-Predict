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