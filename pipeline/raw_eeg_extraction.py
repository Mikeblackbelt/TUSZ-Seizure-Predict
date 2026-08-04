import mne
import numpy as np

from pipeline.checkpoint_io import save_checkpoint
from pipeline.eeg_channels import CHANNELS_TO_INCLUDE, N_TARGET_CHANNELS
from util import handle_logs

logger = handle_logs.get_logger("raw_eeg_extraction", "applog")

TARGET_SFREQ = 256  # Define target sampling rate centrally


def concatenate_session_eeg(session, session_key=None, output_dir=None):
    """
    Reads a session's .edf files, resamples each directly to TARGET_SFREQ (256 Hz),
    and concatenates them into a continuous (N_TARGET_CHANNELS, total_resampled_samples) array.

    Returns:
        np.ndarray of shape (N_TARGET_CHANNELS, total_resampled_samples), or None if
        the session has no .edf files.
    """
    edf_paths = session.get("edf_paths", [])

    if not edf_paths:
        logger.warning("Session has no .edf files - nothing to concatenate")
        return None

    logger.info(f"Processing and resampling {len(edf_paths)} .edf recordings to {TARGET_SFREQ} Hz")

    resampled_chunks = []

    for edf_path in edf_paths:
        raw = mne.io.read_raw_edf(edf_path, include=CHANNELS_TO_INCLUDE, verbose="Error")

        if len(raw.ch_names) != N_TARGET_CHANNELS:
            raise ValueError(
                f"{edf_path}: found {len(raw.ch_names)} target channels, "
                f"expected {N_TARGET_CHANNELS}"
            )

        if raw.info["sfreq"] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ, npad="auto", verbose="Error")

        data = raw.get_data()
        resampled_chunks.append(data)
        logger.debug(f"Read & resampled {edf_path} to shape {data.shape}")

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
        save_checkpoint(combined, session_key, output_dir, stage="raw")

    return combined

#test
if __name__ == "__main__":
    from pipeline.session_index import index_sessions
    from pipeline.raw_eeg_extraction import concatenate_session_eeg
    import time

    sessions = index_sessions("dev")
    session_keys = list(sessions.keys())[:10]

    for key in session_keys:
        session = sessions[key]

        print(f"\n at key {key}")
        print(f"  {len(session['edf_paths'])} .edf files")

        start_time = time.time()

        result = concatenate_session_eeg(
            session,
            session_key=key,
            output_dir="raweeg_output"
        )

        elapsed = time.time() - start_time

        if result is not None:
            print(f"\nShape: {result.shape}")
            print(f"Saved to: raweeg_output/{key}_raw.npy")
            print(f"Time consumed: {elapsed:.2f} seconds")
        else:
            print("No .edf files in this session")