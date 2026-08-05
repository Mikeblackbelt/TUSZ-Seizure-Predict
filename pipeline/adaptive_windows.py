import numpy as np


def do_overlap_np(original_seq, seg_len, overlap_len):
    """Slices a 2D NumPy array [channels, seq_len] into fixed-size windows."""
    res = []
    step = seg_len - overlap_len
    for i in range(0, original_seq.shape[1], step):
        if original_seq.shape[1] - i < seg_len:
            break
        res.append(original_seq[:, i : (i + seg_len)])
    return res


def get_adaptive_windows_npy(
    pre_files, inter_files, seg_time=4, sfreq=256, overlap=True
):
    """Loads pre-ictal and inter-ictal .npy files and applies adaptive sliding windows.

    Args:
        pre_files (list of str): File paths to pre-ictal .npy files [channels,
          time_samples]
        inter_files (list of str): File paths to inter-ictal .npy files
          [channels, time_samples]
        seg_time (float/int): Window size in seconds (default: 4s)
        sfreq (int): Sampling frequency in Hz (default: 256Hz)
        overlap (bool): Whether to use dynamic overlap on pre-ictal data

    Returns:
        segments_inter_no_overlap (list of np.ndarray): Non-overlapping
        inter-ictal windows
        segments_pre_overlap (list of np.ndarray): Adaptively overlapped
        pre-ictal windows
    """
    seg_len = int(seg_time * sfreq)

    segments_pre = [np.load(f) for f in pre_files]
    segments_inter = [np.load(f) for f in inter_files]

    total_len_pre = sum(arr.shape[1] for arr in segments_pre)
    total_len_inter = sum(arr.shape[1] for arr in segments_inter)

    if total_len_pre == 0 or total_len_inter == 0:
        raise ValueError(
            "Pre-ictal or inter-ictal data is empty. Check your file paths."
        )

    # Compute dynamic step size to balance sample counts
    step_for_overlap = int(seg_len / (total_len_inter / total_len_pre))
    step_for_overlap = max(1, step_for_overlap)  # Prevent zero-division/infinite loop

    segments_pre_overlap = []
    segments_inter_no_overlap = []

    # Extract windows
    while segments_inter or segments_pre:
        if segments_pre:
            if overlap:
                overlap_len = seg_len - step_for_overlap
                segments_pre_overlap.extend(
                    do_overlap_np(segments_pre[0], seg_len, overlap_len)
                )
            else:
                segments_pre_overlap.extend(
                    do_overlap_np(segments_pre[0], seg_len, 0)
                )
            segments_pre.pop(0)

        if segments_inter:
            segments_inter_no_overlap.extend(
                do_overlap_np(segments_inter[0], seg_len, 0)
            )
            segments_inter.pop(0)

    return segments_inter_no_overlap, segments_pre_overlap