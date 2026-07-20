import os

from pipeline.preictal_segment import get_split
from util import handle_logs

logger = handle_logs.get_logger("session_index", "applog")

_SPLIT_PREFIX = {"train": "trn", "dev": "vld", "eval": "tst"}


def _parse_montage_type(recording_folder):
    """
    Derive a short montage-type tag (e.g. 'ar1', 'le2') from a recording
    folder name like '01_tcp_ar' or '03_tcp_ar_a'.
    """
    lower = recording_folder.lower()
    prefix = recording_folder.split("_")[0]
    number = str(int(prefix)) if prefix.isdigit() else "0"

    if "tcp_ar" in lower:
        ref = "ar"
    elif "tcp_le" in lower:
        ref = "le"
    else:
        ref = "unk"

    return f"{ref}{number}"


def index_sessions(dataset_path):
    """
    Returns:
        dict: Mapping of session_key (the composed ID, e.g.
            "trn_p001_s001_2015_ar1") -> {
            "split": str,
            "patient_id": str,
            "session_id": str,   # raw TUSZ session folder name, e.g. "s001_2015"
            "montage_type": str,
            "edf_paths": list[str],
            "csv_paths": list[str],      # per-channel annotation files
            "csv_bi_paths": list[str],   # aggregated binary annotation files
        }
    """
    logger.info(f"Indexing sessions under {dataset_path}")
    sessions = {}
    skipped_unknown_split = 0
    skipped_unexpected_depth = 0

    for root, _dirs, files in os.walk(dataset_path):
        edf_files = sorted(f for f in files if f.endswith(".edf"))
        csv_bi_files = sorted(f for f in files if f.endswith(".csv_bi"))
        csv_files = sorted(f for f in files if f.endswith(".csv"))

        if not edf_files and not csv_bi_files and not csv_files: 
            continue

        split = get_split(root)
        parts = os.path.normpath(root).split(os.sep)
        parts_lower = [p.lower() for p in parts]

        if split == "unknown" or split not in parts_lower:
            logger.warning(f"Skipping {root}: could not determine split")
            skipped_unknown_split += 1
            continue

        split_idx = parts_lower.index(split)

        try:
            patient_id = parts[split_idx + 1]
            session_id = parts[split_idx + 2]
            recording_folder = parts[split_idx + 3]
        except IndexError:
            logger.warning(
                f"Skipping {root}: unexpected directory depth under split '{split}' "
                f"(expected <split>/<patient>/<session>/<recording>)"
            )
            skipped_unexpected_depth += 1
            continue

        montage_type = _parse_montage_type(recording_folder)
        split_prefix = _SPLIT_PREFIX.get(split, split)
        session_key = f"{split_prefix}_{patient_id}_{session_id}_{montage_type}"

        #build unique identifier
        edf_paths = [os.path.join(root, f) for f in edf_files]
        csv_paths = [os.path.join(root, f) for f in csv_files]
        csv_bi_paths = [os.path.join(root, f) for f in csv_bi_files]

        if session_key not in sessions:
            sessions[session_key] = {
                "split": split,
                "patient_id": patient_id,
                "session_id": session_id,
                "montage_type": montage_type,
                "edf_paths": [],
                "csv_paths": [],
                "csv_bi_paths": [],
            }

        sessions[session_key]["edf_paths"].extend(edf_paths)
        sessions[session_key]["csv_paths"].extend(csv_paths)
        sessions[session_key]["csv_bi_paths"].extend(csv_bi_paths)

    for record in sessions.values():
        record["edf_paths"].sort()
        record["csv_paths"].sort()
        record["csv_bi_paths"].sort()

    logger.info(
        f"Indexed {len(sessions)} sessions "
        f"(skipped {skipped_unknown_split} unknown-split dirs, "
        f"{skipped_unexpected_depth} unexpected-depth dirs)"
    )
    return sessions

session = index_sessions("dev")