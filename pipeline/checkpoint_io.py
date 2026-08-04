import os

import numpy as np

from util import handle_logs

logger = handle_logs.get_logger("checkpoint_io", "applog")

_VALID_STAGES = ("raw", "proc")


def save_checkpoint(array: np.ndarray, session_key: str, output_dir: str, stage: str) -> str:
    """
    Save a session's EEG data array to a .npy file for later retrieval.
    """
    if stage not in _VALID_STAGES:
        raise ValueError(f"stage must be one of {_VALID_STAGES}, got {stage!r}")
    if not session_key:
        raise ValueError("session_key is required to build the output filename")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{session_key}_{stage}.npy")
    np.save(out_path, array)
    logger.info(f"Saved {stage} checkpoint for {session_key} to {out_path} (shape={array.shape})")
    return out_path


def load_checkpoint(session_key: str, output_dir: str, stage: str) -> np.ndarray:
    
    if stage not in _VALID_STAGES:
        raise ValueError(f"stage must be one of {_VALID_STAGES}, got {stage!r}")

    path = os.path.join(output_dir, f"{session_key}_{stage}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {stage} checkpoint found for {session_key} at {path}")

    array = np.load(path)
    logger.info(f"Loaded {stage} checkpoint for {session_key} from {path} (shape={array.shape})")
    return array