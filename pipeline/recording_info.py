import mne

from pipeline.eeg_channels import CHANNELS_TO_INCLUDE, N_TARGET_CHANNELS
from util import handle_logs

logger = handle_logs.get_logger("recording_info", "applog")


def get_recording_info(edf_path):
    """
    Read header-level metadata from a single .edf recording, restricted to
    the pipeline's 17 target channels.

    Returns:
        dict: {
            "sfreq": float,          # sampling rate, Hz
            "n_times": int,          # number of samples per channel
            "ch_names": list[str],   # the target channels actually found
        }
    """
    raw = mne.io.read_raw_edf(edf_path, include=CHANNELS_TO_INCLUDE, verbose="Warning")

    info = {
        "sfreq": raw.info["sfreq"],
        "n_times": raw.n_times,
        "ch_names": list(raw.ch_names),
    }

    if len(info["ch_names"]) != N_TARGET_CHANNELS:
        logger.warning(
            f"{edf_path}: found {len(info['ch_names'])} target channels "
            f"(expected {N_TARGET_CHANNELS}): {info['ch_names']}"
        )

    return info