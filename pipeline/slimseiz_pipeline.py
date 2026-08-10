from util import handle_logs
from pipeline.session_index import index_sessions     
from pipeline.raw_eeg_extraction import concatenate_session_eeg
from pipeline.session_metadata import extract_session_metadata
import time
import os
import pandas as pd
import numpy as np
from pathlib import Path
from pipeline.segment_npy import extract_segments
from pipeline.windows import segment_fixed, segment_adaptive
from pipeline.slimseiz_model import OneDCNN
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

logger = handle_logs.get_logger("slimseiz_pipeline", "applog")
def runSlimSeizPreprocessing():
    #index sessions
    indexed_sessions = index_sessions("train")
    MAX_SESSIONS = 50
    session_keys = list(indexed_sessions.keys())[:MAX_SESSIONS]
    indexed_sessions = {k: indexed_sessions[k] for k in session_keys}

    #concatenate sessions (drops channels, resamples, and saves to .npy)
    
    for key, session in indexed_sessions.items():
        session = indexed_sessions[key]

        logger.info(f"\n at key {key}")
        logger.info(f"  {len(session['edf_paths'])} .edf files")

        start_time = time.time()

        result, file_offsets = concatenate_session_eeg(
            session,
            session_key=key,
            output_dir="raweeg_output"
        )

        elapsed = time.time() - start_time

        if result is not None:
            logger.info(f"\nShape: {result.shape}")
            logger.info(f"Saved to: raweeg_output/{key}_raw.npy")
            logger.info(f"Time consumed: {elapsed:.2f} seconds")
        else:
            logger.warning("No .edf files in this session")

        #session metadata 
        metadata = extract_session_metadata(session)
        logger.info(f"Metadata for {key}: {metadata}")

        extract_session_metadata(session)

    master_df = pd.read_csv("master_full.csv")

    # Valid preictal windows only - status=1, exclude the sopbuffer rows,
    # which mark the SOP safety gap, not extractable preictal signal.
    preictal_labels = [l for l in master_df["label"].unique()
                        if l.startswith("p") and not l.endswith("_sopbuffer")]
    preictal_windows = extract_segments(
        master_df, indexed_sessions, output_dir="raweeg_output",
        label_filter=preictal_labels
    )

    interictal_windows = extract_segments(
        master_df, indexed_sessions, output_dir="raweeg_output",
        label_filter=["interictal"]
    )
    from pipeline.bipolar_montages import BIPOLAR_PAIRS, channel_index_dict
    from pipeline.eeg_channels import N_TARGET_CHANNELS

    def montage_windows(windows):
        # note: this is basically doing what create_bipolar_montages is, but that function makes us load and save
        # the files from disk. i would change the function but the others are using it, so i might break their code if i change it
        montaged = []
        for w in windows:
            arr = w["segment"]
            if not (arr.shape[0] == N_TARGET_CHANNELS and arr.shape[1] > 0):
                logger.warning(f"Skipping window {w.get('label')} — bad segment shape {arr.shape}")
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

    SEG_TIME = 4.0
    SFREQ = 256

    # Fixed (non-overlapping) segmentation for the majority class
    interictal_segments = segment_fixed(interictal_windows, SEG_TIME, SFREQ)

    # Adaptive (overlapping) segmentation for the minority class,
    # balanced against interictal's total sample count
    total_len_inter = sum(w["window"].shape[1] for w in interictal_windows)
    preictal_segments = segment_adaptive(
        preictal_windows, SEG_TIME, SFREQ, total_len_inter=total_len_inter
    )

    Path("preictal").mkdir(exist_ok=True)
    for i, s in enumerate(preictal_segments):
        np.save(f"preictal/{s['label']}_{i}.npy", s["segment"])  

    Path("interictal").mkdir(exist_ok=True)
    for i, s in enumerate(interictal_segments):
        np.save(f"interictal/{s['label']}_{i}.npy", s["segment"])
    #normalize

class EEGSegmentDataset(Dataset):
    def __init__(self, preictal_dir="preictal", interictal_dir="interictal", normalize=True):
        self.files = []
        self.labels = []

        for f in os.listdir(preictal_dir):
            self.files.append(os.path.join(preictal_dir, f))
            self.labels.append(1)  # preictal = 1

        for f in os.listdir(interictal_dir):
            self.files.append(os.path.join(interictal_dir, f))
            self.labels.append(0)  # interictal = 0

        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])  # shape (20, 1024)

        if self.normalize:
            arr = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-8)

        x = torch.tensor(arr, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
if __name__ == "__main__":
    
    # runSlimSeizPreprocessing()
    model = OneDCNN(input_channels=20) #bipolar montages gives 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = EEGSegmentDataset("preictal", "interictal")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        print(f"Epoch {epoch+1}: loss={total_loss/total:.4f}, acc={correct/total:.4f}")