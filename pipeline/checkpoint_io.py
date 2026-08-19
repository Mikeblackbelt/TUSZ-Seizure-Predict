import os
import json
import numpy as np

from util import handle_logs

logger = handle_logs.get_logger("checkpoint_io", "applog")

_VALID_STAGES = ("raw", "proc")


def _checkpoint_path(output_dir: str, session_key: str, stage: str) -> str:
    return os.path.join(output_dir, f"{session_key}_{stage}.npz")


def _prepare_array_for_storage(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype.kind == "f":
        return array.astype(np.float32, copy=False)
    return array


def save_checkpoint(array: np.ndarray, session_key: str, output_dir: str, stage: str) -> str:
    """
    Save a session's EEG data array to a compressed .npz file for later retrieval.
    """
    if stage not in _VALID_STAGES:
        raise ValueError(f"stage must be one of {_VALID_STAGES}, got {stage!r}")
    if not session_key:
        raise ValueError("session_key is required to build the output filename")

    os.makedirs(output_dir, exist_ok=True)
    out_path = _checkpoint_path(output_dir, session_key, stage)
    tmp_path = out_path + ".tmp"
    compact_array = _prepare_array_for_storage(array)
    np.savez_compressed(tmp_path, data=compact_array)
    os.replace(tmp_path, out_path)
    logger.info(
        f"Saved {stage} checkpoint for {session_key} to {out_path} "
        f"(shape={compact_array.shape}, dtype={compact_array.dtype})"
    )
    return out_path


def load_checkpoint(session_key: str, output_dir: str, stage: str) -> np.ndarray:
    
    if stage not in _VALID_STAGES:
        raise ValueError(f"stage must be one of {_VALID_STAGES}, got {stage!r}")

    path = _checkpoint_path(output_dir, session_key, stage)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {stage} checkpoint found for {session_key} at {path}")

    with np.load(path) as data:
        array = data["data"]
    logger.info(f"Loaded {stage} checkpoint for {session_key} from {path} (shape={array.shape})")
    return array

def save_offsets(file_offsets: list, session_key: str, output_dir: str) -> str:
    """
    Save a session's per-file sample offsets alongside its raw checkpoint.
    """
    if not session_key:
        raise ValueError("session_key is required to build the output filename")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{session_key}_offsets.json")
    with open(out_path, "w") as f:
        json.dump(file_offsets, f, indent=2)
    logger.info(f"Saved offsets for {session_key} to {out_path} ({len(file_offsets)} files)")
    return out_path


def load_offsets(session_key: str, output_dir: str) -> list:
    path = os.path.join(output_dir, f"{session_key}_offsets.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No offsets found for {session_key} at {path}")

    with open(path) as f:
        file_offsets = json.load(f)
    logger.info(f"Loaded offsets for {session_key} from {path} ({len(file_offsets)} files)")
    return file_offsets