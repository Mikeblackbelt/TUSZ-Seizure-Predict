import os

import mne
import numpy as np

from pipeline.eeg_channels import CHANNELS_TO_INCLUDE, N_TARGET_CHANNELS
from pipeline.recording_info import get_recording_info
from util import handle_logs

logger = handle_logs.get_logger("raw_eeg_extraction", "applog")

def concatenate_session_eeg(session, session_key=None, output_dir=None):
    """
    Concatenate a session's .edf recordings (in order) into one continuous
    (N_TARGET_CHANNELS, total_samples) raw EEG array.

    Returns:
        np.ndarray of shape (N_TARGET_CHANNELS, total_samples), or None if
        the session has no .edf files.
    """
    edf_paths = session.get("edf_paths", [])

    if not edf_paths:
        logger.warning("Session has no .edf files - nothing to concatenate")
        return None

    logger.info(f"Reading recording info for {len(edf_paths)} recordings")
    recording_infos = [get_recording_info(path) for path in edf_paths]

    sample_counts = [info["n_times"] for info in recording_infos]
    total_samples = sum(sample_counts)

    #running total
    offsets = [0]
    for count in sample_counts:
        offsets.append(offsets[-1] + count)

    #reserve space in array for concatenated data
    combined = np.zeros((N_TARGET_CHANNELS, total_samples))

    for i, edf_path in enumerate(edf_paths):
        raw = mne.io.read_raw_edf(edf_path, include=CHANNELS_TO_INCLUDE, verbose="Warning")
        data = raw.get_data()

        expected_n_times = recording_infos[i]["n_times"]
        if data.shape[1] != expected_n_times:
            raise ValueError(
                f"{edf_path}: sample count changed between get_recording_info() "
                f"({expected_n_times}) and get_data() ({data.shape[1]})"
            )
        if data.shape[0] != N_TARGET_CHANNELS:
            raise ValueError(
                f"{edf_path}: found {data.shape[0]} target channels, "
                f"expected {N_TARGET_CHANNELS}"
            )

        start, end = offsets[i], offsets[i + 1]
        combined[:, start:end] = data
        logger.debug(f"Wrote {edf_path} into samples [{start}:{end}]")

    logger.info(
        f"Concatenated {len(edf_paths)} recordings into "
        f"({N_TARGET_CHANNELS}, {total_samples}) array"
    )

    if output_dir is not None:
        if not session_key:
            raise ValueError(
                "output_dir was given but session_key was not - "
                "cannot determine output filename"
            )
        #Saving to disk is optional
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{session_key}.npy")
        np.save(out_path, combined)
        logger.info(f"Saved concatenated EEG to {out_path}")

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
            print(f"Saved to: raweeg_output/{key}.npy")
            print(f"Time consumed: {elapsed:.2f} seconds")
        else:
            print("No .edf files in this session")
