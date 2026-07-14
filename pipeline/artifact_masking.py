"""
  - apply_zero_masking          -> zero out flagged windows
  - apply_interpolation_masking -> replace flagged windows with a linear interpolation across the surrounding clean samples, per channel

ArtifactDetector.predict_channel resamples each channel to 256Hz and segments into 512-sample windows -- i.e. each window = 512 / 256 = 2.0 seconds, regardless of native fs. 
To act on the *correct* stretch of the *native-rate* signal, we recompute the window length in native samples (round(2.0 * fs_native)) rather than assuming 512 directly, since 512 only applies at 256Hz.

Windows are treated as contiguous and non-overlapping in the same order ArtifactDetector produces them (matches _segment_into_windows, which does sig[:n].reshape(-1, 512) with no overlap).
"""

import numpy as np

WINDOW_SECONDS = 2 # = 2.0s, fixed by ArtifactDetector's internal 256Hz/512-sample windowing


def build_artifact_mask(per_channel_probs, n_channels, n_samples_native,
                         fs_native, artifact_classes=(1, 2)):
    """
    Returns a boolean mask, shape (n_channels, n_samples_native),
    True = flagged as artifact.

    per_channel_probs: list of arrays (one per channel), each shape (n_windows, n_classes), what ArtifactDetector.predict_segment()["per_channel_probs"] returns.
    artifact_classes: which predicted class indices count as "artifact". 
    """
    mask = np.zeros((n_channels, n_samples_native), dtype=bool)
    win_len_native = int(round(WINDOW_SECONDS * fs_native))

    for ch_idx, probs in enumerate(per_channel_probs):
        if len(probs) == 0:
            continue
        pred = probs.argmax(axis=1)
        for w, cls in enumerate(pred):
            if cls not in artifact_classes:
                continue
            start = w * win_len_native
            if start >= n_samples_native:
                break
            end = min(start + win_len_native, n_samples_native)
            mask[ch_idx, start:end] = True

    return mask


def apply_zero_masking(data, detector_result, fs_native, artifact_classes=(1, 2)):
    """
    Returns (masked_data, mask): flagged windows zeroed out.
    Keep `mask` around to also exclude those samples from a loss function, zeros alone can look like flat signal to a downstream model.
    """
    n_channels, n_samples = data.shape
    mask = build_artifact_mask(
        detector_result["per_channel_probs"],
        n_channels, n_samples, fs_native,
        artifact_classes=artifact_classes,
    )
    masked_data = data.copy()
    masked_data[mask] = 0.0
    return masked_data, mask


def apply_interpolation_masking(data, detector_result, fs_native, artifact_classes=(1, 2)):
    """
    Returns (interpolated_data, mask, fully_flagged_channels):

    For each channel, flagged samples are replaced with a linear interpolation across the nearest clean samples on either side (np.interp). 
    Samples before the first clean sample or after the last clean sample get the nearest clean value held constant rather than extrapolated.

    If an entire channel has no clean samples at all (fully_flagged), that channel is left untouched in the output and its index is returned in fully_flagged_channels so you can decide how to handle it (e.g. drop the channel, or fallback to zeroing for that channel specifically).
    """
    n_channels, n_samples = data.shape
    mask = build_artifact_mask(
        detector_result["per_channel_probs"],
        n_channels, n_samples, fs_native,
        artifact_classes=artifact_classes,
    )

    interpolated = data.copy()
    fully_flagged_channels = []
    all_idx = np.arange(n_samples)

    for ch in range(n_channels):
        ch_mask = mask[ch]
        if not ch_mask.any():
            continue  # nothing flagged on this channel

        clean_idx = all_idx[~ch_mask]
        if len(clean_idx) == 0:
            fully_flagged_channels.append(ch)
            continue

        flagged_idx = all_idx[ch_mask]
        interpolated[ch, flagged_idx] = np.interp(
            flagged_idx, clean_idx, data[ch, clean_idx]
        )

    return interpolated, mask, fully_flagged_channels


if __name__ == "__main__":
    n_channels, n_samples, fs_native = 2, 5120, 512
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_channels, n_samples)).astype(np.float64)

    # Fake detector_result: 10 windows/channel, alternating clean/artifact.
    n_windows = 10
    fake_probs = []
    for ch in range(n_channels):
        probs = np.zeros((n_windows, 3))
        for w in range(n_windows):
            cls = w % 3
            probs[w, cls] = 1.0
        fake_probs.append(probs)
    fake_result = {"per_channel_probs": fake_probs, "artifact_fraction": 0.5, "total_windows": n_windows}

    # Zeroing self-test
    masked, mask = apply_zero_masking(data, fake_result, fs_native)
    assert masked.shape == data.shape
    assert (masked[mask] == 0.0).all()
    print(f"[zero] flagged {mask.sum()} / {mask.size} samples ({100*mask.mean():.1f}%)")

    # Interpolation self-test
    interp, mask2, fully_flagged = apply_interpolation_masking(data, fake_result, fs_native)
    assert interp.shape == data.shape
    assert np.array_equal(mask, mask2)
    # Flagged samples should no longer equal the noisy original (with overwhelming probability)
    assert not np.allclose(interp[mask2], data[mask2])
    # Unflagged samples must be untouched
    assert np.array_equal(interp[~mask2], data[~mask2])
    print(f"[interp] replaced {mask2.sum()} samples via interpolation; "
          f"fully-flagged channels: {fully_flagged}")
    print("Self-test passed.")