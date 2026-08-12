from util import handle_logs
from pipeline.session_index import index_sessions     
from pipeline.raw_eeg_extraction import concatenate_session_eeg
from pipeline.session_metadata import extract_session_metadata
from pipeline.segment_npy import extract_segments
from pipeline.windows import segment_fixed
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

# ------------------ FEATURE EXTRACTION ------------------

processed_data_dir_pre = "preictal"
processed_data_dir_inter = "interictal"
feature_output_dir = "processed_data/features"

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

def run_feature_extraction():
    # Make sure the output directory exists first
    Path(feature_output_dir).mkdir(parents=True, exist_ok=True)

    # ---------------- PREICTAL ----------------
    pre_files = glob.glob(os.path.join(processed_data_dir_pre, "*.npy"))
    for file in pre_files:
        base = os.path.basename(file)
        out_path = os.path.join(feature_output_dir, f"preictal_{base}")

        # The short-circuit check for individual files
        if os.path.exists(out_path):
            print(f"Skipping preictal: {base} (Features already exist)")
            continue

        print("Processing preictal:", file)
        try:
            data = np.load(file)  # shape (channels, samples)
            features = gen_time_domain_features(data)
            np.save(out_path, features)
        except Exception as e:
            print("Failed (preictal):", file)
            print(e)

    # ---------------- INTERICTAL ----------------
    inter_files = glob.glob(os.path.join(processed_data_dir_inter, "*.npy"))
    for file in inter_files:
        base = os.path.basename(file)
        out_path = os.path.join(feature_output_dir, f"interictal_{base}")

        # The short-circuit check for individual files
        if os.path.exists(out_path):
            print(f"Skipping interictal: {base} (Features already exist)")
            continue

        print("Processing interictal:", file)
        try:
            data = np.load(file)  # shape (channels, samples)
            features = gen_time_domain_features(data)
            np.save(out_path, features)
        except Exception as e:
            print("Failed (interictal):", file)
            print(e)

# ------------------ PCA + MAT EXPORT ------------------

def run_pca_and_export_mat():
    files = sorted(glob.glob(os.path.join(feature_output_dir, "*.npy")))
    all_features = []
    labels = []

    for f in files:
        vec = np.load(f)  # (176,)
        all_features.append(vec)

        # label from filename prefix
        base = os.path.basename(f)
        if base.startswith("preictal_"):
            labels.append(1)
        elif base.startswith("interictal_"):
            labels.append(0)
        else:
            # default: interictal
            labels.append(0)

    all_features = np.vstack(all_features)  # (N, 176)
    labels = np.array(labels)

    print("All features shape:", all_features.shape)

    pca = PCA(n_components=64)
    compressed = pca.fit_transform(all_features)  # (N, 64)

    # MATLAB expects (64 x N)
    test_data = compressed.T
    test_labels = labels
    n_samples = test_data.shape[1]

    savemat("test_inputs.mat", {
        "test_data": test_data,
        "test_labels": test_labels,
        "n_samples": n_samples
    })

    print("Saved test_inputs.mat with shape:", test_data.shape)

# ------------------ MAIN PIPELINE ------------------

if __name__ == "__main__":
    final_output_file = "test_inputs.mat"

    # Top-level short-circuit: Skips the whole pipeline if the final file is done
    if os.path.exists(final_output_file):
        logger.info(f"Final output '{final_output_file}' already exists. Skipping pipeline.")
        print(f"✅ '{final_output_file}' found! Skipping data extraction and processing.")
        sys.exit(0) 

    print(f"'{final_output_file}' not found. Starting the pipeline...")
    
    indexed_sessions = index_sessions("train")
    MAX_SESSIONS = 50
    session_keys = list(indexed_sessions.keys())[:MAX_SESSIONS]
    indexed_sessions = {k: indexed_sessions[k] for k in session_keys}

    session_data = {}
    for key, session in indexed_sessions.items():
        logger.info(f"\n at key {key}")
        logger.info(f"  {len(session['edf_paths'])} .edf files")

        start_time = time.time()

        result, file_offsets = concatenate_session_eeg(session)

        elapsed = time.time() - start_time

        if result is not None:
            session_data[key] = (result, file_offsets)
            logger.info(f"\nShape: {result.shape}")
            logger.info(f"Time consumed: {elapsed:.2f} seconds")
        else:
            logger.warning("No .edf files in this session")

        metadata = extract_session_metadata(session)
        logger.info(f"Metadata for {key}: {metadata}")

    master_df = pd.read_csv("master_full.csv")

    preictal_labels = [
        l for l in master_df["label"].unique()
        if l.startswith("p") and not l.endswith("_sopbuffer")
    ]

    preictal_windows = extract_segments(
        master_df, indexed_sessions, session_data,
        label_filter=preictal_labels
    )

    interictal_windows = extract_segments(
        master_df, indexed_sessions, session_data,
        label_filter=["interictal"]
    )

    from pipeline.bipolar_montages import BIPOLAR_PAIRS, channel_index_dict
    from pipeline.eeg_channels import N_TARGET_CHANNELS

    def montage_windows(windows):
        montaged = []
        for w in windows:
            arr = w["segment"]
            if not (arr.shape[0] == N_TARGET_CHANNELS and arr.shape[1] > 0):
                logger.warning(
                    f"Skipping window {w.get('label')} — bad segment shape {arr.shape}"
                )
                continue
            proc = np.zeros((len(BIPOLAR_PAIRS), arr.shape[1]))
            for i, (ch1, ch2) in enumerate(BIPOLAR_PAIRS):
                proc[i, :] = arr[channel_index_dict[ch1], :] - arr[channel_index_dict[ch2], :]
            d = {k: v for k, v in w.items() if k != "segment"}
            d["window"] = proc
            montaged.append(d)
        return montaged

    preictal_windows = montage_windows(preictal_windows)
    interictal_windows = montage_windows(interictal_windows)

    SEG_TIME = 64.0
    SFREQ = 256

    interictal_segments = segment_fixed(interictal_windows, SEG_TIME, SFREQ)
    preictal_segments = segment_fixed(
        preictal_windows, SEG_TIME, SFREQ, step_time=SEG_TIME / 2
    )

    Path("preictal").mkdir(exist_ok=True)
    for i, s in enumerate(preictal_segments):
        np.save(f"preictal/{s['label']}_{i}.npy", s["segment"])

    Path("interictal").mkdir(exist_ok=True)
    for i, s in enumerate(interictal_segments):
        np.save(f"interictal/{s['label']}_{i}.npy", s["segment"])

    run_feature_extraction()

    run_pca_and_export_mat()