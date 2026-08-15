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

logger = handle_logs.get_logger("Parallel_CNN", "applog")

# ------------------ CONFIG ------------------

# One master file per split - produced by running segment.py once per
# split (e.g. `python segment.py train --save-config`, then again for
# dev). If you only have a single combined master_full.csv, point both
# splits at it via MASTER_FILE_OVERRIDE.
SPLITS = ["train", "dev"]
MASTER_FILE_TEMPLATE = "master_{split}.csv"
MASTER_FILE_OVERRIDE = None  # e.g. "master_full.csv" to force one file for all splits

MAX_SESSIONS = {"train": None, "dev": None}  # None to disable the cap for a split

SEG_TIME = 64.0
SFREQ = 256

FEATURE_OUTPUT_DIR = "processed_data/features"

# MATLAB side (experiment_runs_low_res.m) loads these via fscanf with a
# hard-coded shape of [64, N] - not a .mat file. testdata.txt is
# 64 x N whitespace-separated floats, testlabel.txt is 1 x N.
# Label convention matches that loader: 0=preictal, 1=interictal.
TESTDATA_FILE = "MATLAB_simulations/testPytorch/testdata.txt"
TESTLABEL_FILE = "MATLAB_simulations/testPytorch/testlabel.txt"

N_PCA_COMPONENTS = 64  # hard requirement - the MATLAB loader assumes exactly 64 rows
NORM_RANGE = (-1.2, 1.2)  # matches the expected input range before quantization


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
    Runs one split (train or dev) end to end, one session at a time:
    concatenate a single session's raw EEG in memory -> extract that
    session's preictal/interictal windows -> bipolar montage -> fixed
    segmentation -> save to preictal_{split}/ and interictal_{split}/ ->
    discard the raw array -> move to the next session.

    This intentionally never holds more than one session's raw array in
    memory at a time (the previous version built a session_data dict for
    the whole split up front, which meant every session's full raw array
    stayed resident simultaneously - the actual cause of the OOM kill,
    not the per-file duplication inside concatenate_session_eeg).

    Returns (n_preictal_segments, n_interictal_segments) actually written.
    """
    master_path = MASTER_FILE_OVERRIDE or MASTER_FILE_TEMPLATE.format(split=split)
    if not os.path.exists(master_path):
        logger.warning(f"[{split}] Master file {master_path} not found - skipping split")
        return 0, 0

    master_df = pd.read_csv(master_path)

    preictal_labels = [
        l for l in master_df["label"].unique()
        if l.startswith("p") and not l.endswith("_sopbuffer")
    ]

    indexed_sessions = index_sessions(split)
    cap = MAX_SESSIONS.get(split)
    if cap is not None:
        session_keys = list(indexed_sessions.keys())[:cap]
        indexed_sessions = {k: indexed_sessions[k] for k in session_keys}

    preictal_dir = Path(f"preictal_{split}")
    interictal_dir = Path(f"interictal_{split}")
    preictal_dir.mkdir(exist_ok=True)
    interictal_dir.mkdir(exist_ok=True)

    n_preictal_written = 0
    n_interictal_written = 0
    n_preictal_windows_total = 0
    n_interictal_windows_total = 0
    n_sessions_with_data = 0

    for key, session in indexed_sessions.items():
        logger.info(f"[{split}] at key {key}")
        logger.info(f"[{split}]   {len(session['edf_paths'])} .edf files")

        start_time = time.time()
        result, file_offsets = concatenate_session_eeg(session)
        elapsed = time.time() - start_time

        if result is None:
            logger.warning(f"[{split}] No .edf files in session {key}")
            continue

        logger.info(f"[{split}] Shape: {result.shape}")
        logger.info(f"[{split}] Time consumed: {elapsed:.2f} seconds")

        metadata = extract_session_metadata(session)
        logger.info(f"[{split}] Metadata for {key}: {metadata}")

        # Single-session view: extract_segments only needs the one
        # session's entries to resolve this session's edf_path -> array.
        session_only = {key: session}
        session_data_only = {key: (result, file_offsets)}

        preictal_windows = extract_segments(
            master_df, session_only, session_data_only, label_filter=preictal_labels
        )
        interictal_windows = extract_segments(
            master_df, session_only, session_data_only, label_filter=["interictal"]
        )

        # Raw array no longer needed past this point - drop the only
        # references to it before moving on to the next session.
        del result, session_data_only
        n_sessions_with_data += 1

        n_preictal_windows_total += len(preictal_windows)
        n_interictal_windows_total += len(interictal_windows)

        if not preictal_windows and not interictal_windows:
            logger.debug(f"[{split}] {key}: no preictal or interictal windows")
            continue

        preictal_windows = montage_windows(preictal_windows, split)
        interictal_windows = montage_windows(interictal_windows, split)

        interictal_segments = segment_fixed(interictal_windows, SEG_TIME, SFREQ)
        preictal_segments = segment_fixed(
            preictal_windows, SEG_TIME, SFREQ, step_time=SEG_TIME / 2
        )

        for s in preictal_segments:
            np.save(preictal_dir / f"{s['label']}_{n_preictal_written}.npy", s["segment"])
            n_preictal_written += 1
        for s in interictal_segments:
            np.save(interictal_dir / f"{s['label']}_{n_interictal_written}.npy", s["segment"])
            n_interictal_written += 1

    if n_sessions_with_data == 0:
        logger.warning(f"[{split}] No sessions produced usable EEG data - skipping split")
        return 0, 0

    if n_preictal_written == 0:
        logger.warning(f"[{split}] 0 preictal segments written across all sessions")
    if n_interictal_written == 0:
        logger.warning(f"[{split}] 0 interictal segments written across all sessions")

    logger.info(
        f"[{split}] raw windows (pre-segmentation): "
        f"{n_preictal_windows_total} preictal, {n_interictal_windows_total} interictal"
    )
    logger.info(
        f"[{split}] wrote {n_preictal_written} preictal segments to {preictal_dir} "
        f"(from {n_preictal_windows_total} windows), "
        f"{n_interictal_written} interictal segments to {interictal_dir} "
        f"(from {n_interictal_windows_total} windows)"
    )

    return n_preictal_written, n_interictal_written


# ------------------ FEATURE EXTRACTION ------------------

def run_feature_extraction(split):
    """Reads preictal_{split}/ and interictal_{split}/, writes per-file
    feature vectors into FEATURE_OUTPUT_DIR, tagged with split+label so
    run_pca_and_export_matlab_inputs() can recover both from the filename."""
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

def run_pca_and_export_matlab_inputs():
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

        # Label convention matches the MATLAB loader: 0=preictal, 1=interictal.
        if f"{split}_preictal_" in base:
            label = 0
        elif f"{split}_interictal_" in base:
            label = 1
        else:
            logger.warning(f"Skipping feature file with unrecognized label prefix: {base}")
            continue

        all_features.append(np.load(f))  # (160,) for the bipolar montage's 20 channels x 8 stats
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

    all_features = np.vstack(all_features)  # (N, 160)
    labels = np.array(labels)
    splits = np.array(splits)

    print("All features shape:", all_features.shape)

    # Hard requirement: the MATLAB loader assumes exactly 64 rows
    # (testdata = fscanf(..., [64, N])). Silently falling back to fewer
    # components would produce a file that loads into the wrong shape
    # without any error - fail loudly here instead, since "not enough
    # samples yet" is a clearer signal than a corrupted downstream load.
    if all_features.shape[0] < N_PCA_COMPONENTS or all_features.shape[1] < N_PCA_COMPONENTS:
        raise ValueError(
            f"Need at least {N_PCA_COMPONENTS} samples and {N_PCA_COMPONENTS} feature "
            f"dims to fit a {N_PCA_COMPONENTS}-component PCA, got "
            f"{all_features.shape[0]} samples x {all_features.shape[1]} dims. "
            f"The MATLAB simulation expects a fixed [64, N] input - reduce "
            f"N_PCA_COMPONENTS only if you deliberately want to change what the "
            f"MATLAB side reads."
        )

    pca = PCA(n_components=N_PCA_COMPONENTS)
    compressed = pca.fit_transform(all_features)  # (N, 64)

    # MATLAB expects (64 x N)
    features_out = compressed.T

    # Normalize to NORM_RANGE per the reference pipeline, fit on train only
    # to avoid leaking dev-split scale into the training distribution, then
    # applied to the full (train+dev) array before writing.
    train_mask = splits == 0
    train_features = features_out[:, train_mask]
    train_min = train_features.min()
    train_max = train_features.max()
    if train_max == train_min:
        raise ValueError("All train-split PCA features are identical - cannot normalize (zero range)")

    lo, hi = NORM_RANGE
    features_norm = (features_out - train_min) / (train_max - train_min)  # -> [0, 1]
    features_norm = features_norm * (hi - lo) + lo  # -> [lo, hi]

    Path(TESTDATA_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(TESTLABEL_FILE).parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(TESTDATA_FILE, features_norm, fmt="%f")
    np.savetxt(TESTLABEL_FILE, labels.reshape(1, -1), fmt="%f")

    print(f"Saved {TESTDATA_FILE} with shape:", features_norm.shape)
    print(f"Saved {TESTLABEL_FILE} with shape:", labels.reshape(1, -1).shape)
    return True


# ------------------ MAIN PIPELINE ------------------

if __name__ == "__main__":
    if os.path.exists(TESTDATA_FILE) and os.path.exists(TESTLABEL_FILE):
        logger.info(f"Final outputs '{TESTDATA_FILE}'/'{TESTLABEL_FILE}' already exist. Skipping pipeline.")
        print(f"'{TESTDATA_FILE}' and '{TESTLABEL_FILE}' found! Skipping data extraction and processing.")
        sys.exit(0)

    print(f"'{TESTDATA_FILE}' not found. Starting the pipeline...")

    for split in SPLITS:
        n_pre, n_inter = process_split(split)
        logger.info(f"[{split}] done: {n_pre} preictal / {n_inter} interictal segments")

    for split in SPLITS:
        run_feature_extraction(split)

    ok = run_pca_and_export_matlab_inputs()
    if not ok:
        sys.exit(1)