from util import handle_logs
from pipeline.session_index import index_sessions
from pipeline.raw_eeg_extraction import concatenate_session_eeg
from pipeline.session_metadata import extract_session_metadata
from pipeline.segment_npy import extract_segments
from pipeline.windows import segment_fixed
from pipeline.bipolar_montages import BIPOLAR_PAIRS, channel_index_dict
from pipeline.eeg_channels import N_TARGET_CHANNELS
from pathlib import Path
import time
import os
import glob
import math
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
import sys
from scipy.io import savemat

logger = handle_logs.get_logger("slimseiz", "applog")

# ------------------ CONFIG ------------------

# One master file per split - produced by running segment.py once per
# split (e.g. `python segment.py train --save-config`, then again for
# dev). If you only have a single combined master_full.csv, point both
# splits at it via MASTER_FILE_OVERRIDE.
SPLITS = ["train", "dev"]
MASTER_FILE_TEMPLATE = "master_{split}.csv"
MASTER_FILE_OVERRIDE = None  # e.g. "master_full.csv" to force one file for all splits

MAX_SESSIONS = {"train": 50, "dev": 50}  # None to disable the cap for a split

SEG_TIME = 64.0
SFREQ = 256

FEATURE_OUTPUT_DIR = "processed_data/features"
FINAL_OUTPUT_FILE = "test_inputs.mat"


def det_entropy(channel_data):
    z = np.abs(channel_data)
    entropy = 0.0
    for i in range(len(channel_data)):
        if z[i] > 0:
            entropy += z[i] * math.log(z[i], 2)
    return -entropy


def gen_time_domain_features(data):
    features = []
    for channel in range(data.shape[0]):
        channel_data = np.abs(data[channel])

        mean_val = np.mean(channel_data)
        var_val = np.var(channel_data)

        features.append(mean_val)
        features.append(var_val)
        features.append(skew(channel_data))
        features.append(kurtosis(channel_data))
        # coefficient of variation (guard against divide-by-zero)
        if mean_val != 0:
            features.append(math.sqrt(var_val) / mean_val)
        else:
            features.append(0.0)
        features.append(np.mean(np.abs(channel_data - mean_val)))
        features.append(np.sqrt(np.mean(channel_data ** 2)))
        features.append(det_entropy(channel_data))

    return np.array(features)  # shape (176,)


# ------------------ PER-SPLIT EXTRACTION ------------------

def build_session_data(split):
    """Index a split's sessions and concatenate each one's raw EEG in
    memory (no raw checkpoint is written to disk - see raw_eeg_extraction).
    Returns (indexed_sessions, session_data)."""
    indexed_sessions = index_sessions(split)

    cap = MAX_SESSIONS.get(split)
    if cap is not None:
        session_keys = list(indexed_sessions.keys())[:cap]
        indexed_sessions = {k: indexed_sessions[k] for k in session_keys}

    session_data = {}
    for key, session in indexed_sessions.items():
        logger.info(f"[{split}] at key {key}")
        logger.info(f"[{split}]   {len(session['edf_paths'])} .edf files")

        start_time = time.time()
        result, file_offsets = concatenate_session_eeg(session)
        elapsed = time.time() - start_time

        if result is not None:
            session_data[key] = (result, file_offsets)
            logger.info(f"[{split}] Shape: {result.shape}")
            logger.info(f"[{split}] Time consumed: {elapsed:.2f} seconds")
        else:
            logger.warning(f"[{split}] No .edf files in session {key}")

        metadata = extract_session_metadata(session)
        logger.info(f"[{split}] Metadata for {key}: {metadata}")

    return indexed_sessions, session_data


def montage_windows(windows, split):
    montaged = []
    for w in windows:
        arr = w["segment"]
        if not (arr.shape[0] == N_TARGET_CHANNELS and arr.shape[1] > 0):
            logger.warning(
                f"[{split}] Skipping window {w.get('label')} — bad segment shape {arr.shape}"
            )
            continue
        proc = np.zeros((len(BIPOLAR_PAIRS), arr.shape[1]))
        for i, (ch1, ch2) in enumerate(BIPOLAR_PAIRS):
            proc[i, :] = arr[channel_index_dict[ch1], :] - arr[channel_index_dict[ch2], :]
        d = {k: v for k, v in w.items() if k != "segment"}
        d["window"] = proc
        montaged.append(d)
    return montaged


def process_split(split):
    """
    Runs one split (train or dev) end to end: index -> concatenate ->
    extract preictal/interictal windows -> bipolar montage -> fixed
    segmentation -> save to preictal_{split}/ and interictal_{split}/.

    Returns (n_preictal_segments, n_interictal_segments) actually written.
    """
    master_path = MASTER_FILE_OVERRIDE or MASTER_FILE_TEMPLATE.format(split=split)
    if not os.path.exists(master_path):
        logger.warning(f"[{split}] Master file {master_path} not found - skipping split")
        return 0, 0

    master_df = pd.read_csv(master_path)

    indexed_sessions, session_data = build_session_data(split)
    if not session_data:
        logger.warning(f"[{split}] No sessions produced usable EEG data - skipping split")
        return 0, 0

    preictal_labels = [
        l for l in master_df["label"].unique()
        if l.startswith("p") and not l.endswith("_sopbuffer")
    ]

    preictal_windows = extract_segments(
        master_df, indexed_sessions, session_data, label_filter=preictal_labels
    )
    interictal_windows = extract_segments(
        master_df, indexed_sessions, session_data, label_filter=["interictal"]
    )

    if not preictal_windows:
        logger.warning(f"[{split}] extract_segments returned 0 preictal windows")
    if not interictal_windows:
        logger.warning(f"[{split}] extract_segments returned 0 interictal windows")

    preictal_windows = montage_windows(preictal_windows, split)
    interictal_windows = montage_windows(interictal_windows, split)

    interictal_segments = segment_fixed(interictal_windows, SEG_TIME, SFREQ)
    preictal_segments = segment_fixed(
        preictal_windows, SEG_TIME, SFREQ, step_time=SEG_TIME / 2
    )

    preictal_dir = Path(f"preictal_{split}")
    interictal_dir = Path(f"interictal_{split}")
    preictal_dir.mkdir(exist_ok=True)
    interictal_dir.mkdir(exist_ok=True)

    for i, s in enumerate(preictal_segments):
        np.save(preictal_dir / f"{s['label']}_{i}.npy", s["segment"])
    for i, s in enumerate(interictal_segments):
        np.save(interictal_dir / f"{s['label']}_{i}.npy", s["segment"])

    logger.info(
        f"[{split}] wrote {len(preictal_segments)} preictal segments to {preictal_dir}, "
        f"{len(interictal_segments)} interictal segments to {interictal_dir}"
    )

    return len(preictal_segments), len(interictal_segments)


# ------------------ FEATURE EXTRACTION ------------------

def run_feature_extraction(split):
    """Reads preictal_{split}/ and interictal_{split}/, writes per-file
    feature vectors into FEATURE_OUTPUT_DIR, tagged with split+label so
    run_pca_and_export_mat() can recover both from the filename."""
    Path(FEATURE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for label_prefix, src_dir in (("preictal", f"preictal_{split}"), ("interictal", f"interictal_{split}")):
        files = glob.glob(os.path.join(src_dir, "*.npy"))
        for file in files:
            base = os.path.basename(file)
            out_path = os.path.join(FEATURE_OUTPUT_DIR, f"{split}_{label_prefix}_{base}")

            if os.path.exists(out_path):
                print(f"Skipping {split}/{label_prefix}: {base} (features already exist)")
                continue

            print(f"Processing {split}/{label_prefix}:", file)
            try:
                data = np.load(file)  # shape (channels, samples)
                features = gen_time_domain_features(data)
                np.save(out_path, features)
            except Exception as e:
                print(f"Failed ({split}/{label_prefix}):", file)
                print(e)


# ------------------ PCA + MAT EXPORT ------------------

def run_pca_and_export_mat():
    files = sorted(glob.glob(os.path.join(FEATURE_OUTPUT_DIR, "*.npy")))

    all_features = []
    labels = []
    splits = []

    for f in files:
        base = os.path.basename(f)

        split = "train" if base.startswith("train_") else ("dev" if base.startswith("dev_") else None)
        if split is None:
            logger.warning(f"Skipping feature file with unrecognized split prefix: {base}")
            continue

        if f"{split}_preictal_" in base:
            label = 1
        elif f"{split}_interictal_" in base:
            label = 0
        else:
            logger.warning(f"Skipping feature file with unrecognized label prefix: {base}")
            continue

        all_features.append(np.load(f))  # (176,)
        labels.append(label)
        splits.append(0 if split == "train" else 1)

    if not all_features:
        logger.error(
            f"No feature files found under {FEATURE_OUTPUT_DIR} - nothing to PCA/export. "
            "This means extract_segments/montage_windows/segment_fixed produced zero "
            "segments upstream (check the [split] warnings logged above) rather than a "
            "bug in this step."
        )
        print(f"No feature files found under {FEATURE_OUTPUT_DIR} - aborting before PCA.")
        return False

    all_features = np.vstack(all_features)  # (N, 176)
    labels = np.array(labels)
    splits = np.array(splits)

    print("All features shape:", all_features.shape)

    n_components = min(64, all_features.shape[0], all_features.shape[1])
    if n_components < 64:
        logger.warning(
            f"Only {all_features.shape[0]} samples / {all_features.shape[1]} dims available - "
            f"using n_components={n_components} instead of 64"
        )

    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(all_features)  # (N, n_components)

    # MATLAB expects (n_components x N)
    test_data = compressed.T
    test_labels = labels
    test_splits = splits  # 0=train, 1=dev
    n_samples = test_data.shape[1]

    savemat(FINAL_OUTPUT_FILE, {
        "test_data": test_data,
        "test_labels": test_labels,
        "test_splits": test_splits,
        "n_samples": n_samples,
    })

    print(f"Saved {FINAL_OUTPUT_FILE} with shape:", test_data.shape)
    return True


# ------------------ MAIN PIPELINE ------------------

if __name__ == "__main__":
    if os.path.exists(FINAL_OUTPUT_FILE):
        logger.info(f"Final output '{FINAL_OUTPUT_FILE}' already exists. Skipping pipeline.")
        print(f"'{FINAL_OUTPUT_FILE}' found! Skipping data extraction and processing.")
        sys.exit(0)

    print(f"'{FINAL_OUTPUT_FILE}' not found. Starting the pipeline...")

    for split in SPLITS:
        n_pre, n_inter = process_split(split)
        logger.info(f"[{split}] done: {n_pre} preictal / {n_inter} interictal segments")

    for split in SPLITS:
        run_feature_extraction(split)

    ok = run_pca_and_export_mat()
    if not ok:
        sys.exit(1)