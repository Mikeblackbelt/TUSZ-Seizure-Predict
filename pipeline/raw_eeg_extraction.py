import mne
import numpy as np

from pipeline.eeg_channels import CHANNELS_TO_INCLUDE, N_TARGET_CHANNELS, reorder_to_canonical
from filters.adaptive_filters import detect_noise_frequencies, apply_notch_filter
from filters.simple_filters import bandpass_filter_raw
from util import handle_logs

logger = handle_logs.get_logger("raw_eeg_extraction", "applog")

TARGET_SFREQ = 256  # Define target sampling rate centrally


def concatenate_session_eeg(
    session,
    apply_filtering=True,
    notch_variance_threshold=0.20,
    notch_power_percentile=80,
    bandpass_low=0.5,
    bandpass_high=40.0,
):
    """
    Reads a session's .edf files, filters + resamples each directly to
    TARGET_SFREQ (256 Hz), and concatenates them into a continuous
    (N_TARGET_CHANNELS, total_resampled_samples) array.

    Calculates the file offsets for each recording.

    NOTE: this no longer writes a "raw" checkpoint to disk. The returned
    array and offsets are meant to be kept in memory and passed straight
    into downstream processing (e.g. segment_npy.extract_segments) so the
    final product .npy files are the only thing that ever hits disk.

    Filtering (adaptive notch, then Butterworth bandpass) happens per-file,
    at each file's native sample rate, before resampling - cutoffs are
    relative to native fs, so filtering after resampling would change their
    effective frequencies.

    Parameters:
        apply_filtering (bool): If True (default), run notch + bandpass
            filtering on each file before resampling. Set False to skip
            filtering entirely (e.g. for comparing filtered vs unfiltered
            in-memory arrays).
        notch_variance_threshold, notch_power_percentile: passed through to
            adaptive_filters.detect_noise_frequencies().
        bandpass_low, bandpass_high: passed through to
            simple_filters.bandpass_filter_raw().

    Returns:
        np.ndarray of shape (N_TARGET_CHANNELS, total_resampled_samples), or None if
        the session has no .edf files.
        `file_offsets` (list of dict): List of boundary offsets relative to the session where each dict contains:
            - 'edf_path' (str): Path to the source .edf file.
            - 'start_sample' (int): Inclusive start sample index in `combined`.
            - 'end_sample' (int): Exclusive end sample index in `combined`.
    """
    edf_paths = session.get("edf_paths", [])

    if not edf_paths:
        logger.warning("Session has no .edf files - nothing to concatenate")
        return None, []

    logger.info(
        f"Processing {len(edf_paths)} .edf recordings "
        f"({'filter + ' if apply_filtering else ''}resample to {TARGET_SFREQ} Hz)"
    )

    resampled_chunks = []
    file_offsets = []
    running_sample = 0
    for edf_path in edf_paths:
        # preload=True: both filtering and resample() need a writable
        # in-memory array, not a lazy on-disk reference.
        raw = mne.io.read_raw_edf(edf_path, include=CHANNELS_TO_INCLUDE, preload=True, verbose="Error")

        if len(raw.ch_names) != N_TARGET_CHANNELS:
            raise ValueError(
                f"{edf_path}: found {len(raw.ch_names)} target channels, "
                f"expected {N_TARGET_CHANNELS}"
            )

        if apply_filtering:
            noise_freqs = detect_noise_frequencies(
                raw,
                variance_threshold=notch_variance_threshold,
                power_percentile=notch_power_percentile,
            )
            if noise_freqs:
                logger.debug(f"{edf_path}: notching {noise_freqs}")
                raw = apply_notch_filter(raw, noise_freqs)
            else:
                logger.debug(f"{edf_path}: no notch-worthy frequencies detected")

            raw = bandpass_filter_raw(raw, low_cutoff=bandpass_low, high_cutoff=bandpass_high)

        if raw.info["sfreq"] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ, npad="auto", verbose="Error")

        data = raw.get_data().astype(np.float32)

        # `include=CHANNELS_TO_INCLUDE` selects channels but does NOT
        # guarantee row order - MNE preserves each .edf's own header
        # order, which varies between AR/LE recordings. Enforce a fixed,
        # known row order here so downstream code (bipolar_montages.py)
        # can safely index rows by channel name.
        data = reorder_to_canonical(data, raw.ch_names)

        resampled_chunks.append(data)
        n_samples = data.shape[1]
        file_offsets.append({
            "edf_path": edf_path,
            "start_sample": running_sample,
            "end_sample": running_sample + n_samples,
        })
        running_sample += n_samples
        logger.debug(f"Read & resampled {edf_path} to shape {data.shape}")

    combined = np.concatenate(resampled_chunks, axis=1)

    logger.info(
        f"Successfully concatenated {len(edf_paths)} recordings into "
        f"resampled {combined.shape} array"
    )

    return combined, file_offsets

#test
if __name__ == "__main__":
    from pipeline.session_index import index_sessions
    from pipeline.raw_eeg_extraction import concatenate_session_eeg
    import time

    sessions = index_sessions("train")
    session_keys = list(sessions.keys())[:10]

    for key in session_keys:
        session = sessions[key]

        print(f"\n at key {key}")
        print(f"  {len(session['edf_paths'])} .edf files")

        start_time = time.time()

        result, file_offsets = concatenate_session_eeg(session)

        elapsed = time.time() - start_time

        if result is not None:
            print(f"\nShape: {result.shape}")
            print(f"Time consumed: {elapsed:.2f} seconds")
        else:
            print("No .edf files in this session")