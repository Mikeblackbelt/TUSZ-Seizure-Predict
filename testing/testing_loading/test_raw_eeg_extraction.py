import mne
import numpy as np

from pipeline.checkpoint_io import save_checkpoint, save_offsets
from pipeline.eeg_channels import CHANNELS_TO_INCLUDE, N_TARGET_CHANNELS
from filters.adaptive_filters import detect_noise_frequencies, apply_notch_filter
from filters.simple_filters import bandpass_filter_raw
from util import handle_logs

logger = handle_logs.get_logger("raw_eeg_extraction", "applog")

TARGET_SFREQ = 256 

CANONICAL_CHANNELS = [
    'FP1', 'F7', 'T3', 'T5', 'O1', 'FP2', 'F8', 'T4', 'T6', 'O2', 
    'F3', 'C3', 'P3', 'F4', 'C4', 'P4', 'CZ'
]

def concatenate_session_eeg(
    session,
    session_key=None,
    output_dir=None,
    apply_filtering=True,
    notch_variance_threshold=0.20,
    notch_power_percentile=80,
    bandpass_low=0.5,
    bandpass_high=40.0,
):
    edf_paths = session.get("edf_paths", [])

    if not edf_paths:
        logger.warning("Session has no .edf files - nothing to concatenate")
        return None

    logger.info(
        f"Processing {len(edf_paths)} .edf recordings "
        f"({'filter + ' if apply_filtering else ''}resample to {TARGET_SFREQ} Hz)"
    )

    resampled_chunks = []
    file_offsets = []
    running_sample = 0
    for edf_path in edf_paths:
        raw = mne.io.read_raw_edf(edf_path, include=CHANNELS_TO_INCLUDE, preload=True, verbose="Error")

        if len(raw.ch_names) != N_TARGET_CHANNELS:
            raise ValueError(
                f"{edf_path}: found {len(raw.ch_names)} target channels, "
                f"expected {N_TARGET_CHANNELS}"
            )

        # --- ENFORCE CANONICAL CHANNEL ORDERING ---
        clean_raw_names = [ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '').strip() for ch in raw.ch_names]
        name_to_original = {clean: orig for clean, orig in zip(clean_raw_names, raw.ch_names)}
        
        try:
            ordered_original_names = [name_to_original[clean] for clean in CANONICAL_CHANNELS]
            raw.reorder_channels(ordered_original_names)
        except KeyError as e:
            logger.error(f"Missing expected canonical channel in {edf_path}: {e}")
            continue

        # --- FILTERING ---
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

        # --- RESAMPLING ---
        if raw.info["sfreq"] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ, npad="auto", verbose="Error")

        data = raw.get_data()
        resampled_chunks.append(data)
        n_samples = data.shape[1]
        file_offsets.append({
            "edf_path": edf_path,
            "start_sample": running_sample,
            "end_sample": running_sample + n_samples,
        })
        running_sample += n_samples
        logger.debug(f"Read & resampled {edf_path} to shape {data.shape}")

    if not resampled_chunks:
        return None

    combined = np.concatenate(resampled_chunks, axis=1)

    logger.info(
        f"Successfully concatenated {len(edf_paths)} recordings into "
        f"resampled {combined.shape} array"
    )

    if output_dir is not None:
        if not session_key:
            raise ValueError(
                "output_dir was given but session_key was not - "
                "cannot determine output filename"
            )
        save_offsets(file_offsets, session_key, output_dir)
        save_checkpoint(combined, session_key, output_dir, stage="raw")

    return combined, file_offsets