from util import handle_logs
from pipeline.session_index import index_sessions     
from pipeline.raw_eeg_extraction import concatenate_session_eeg
from pipeline.session_metadata import extract_session_metadata
import time
import os
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from pipeline.segment_npy import extract_segments, build_edf_to_session_map
from pipeline.windows import segment_fixed, segment_adaptive
from pipeline.slimseiz_model import OneDCNN
from pipeline.bipolar_montages import BIPOLAR_PAIRS, channel_index_dict
from pipeline.eeg_channels import N_TARGET_CHANNELS
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

logger = handle_logs.get_logger("slimseiz_pipeline", "applog")
SEG_TIME = 4.0
SFREQ = 256

def montage_windows(windows):
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


def run_pipeline_for_split(split_name, master_csv, raw_output_dir=None):
    logger.info(f"=== Running pipeline for split: {split_name} ===")
    start_time = time.time()
    indexed_sessions = dict(list(index_sessions(split_name).items())[:5])
    master_df = pd.read_csv(master_csv)

    normalized_sessions = {}  # key -> (z-scored combined array, file_offsets)

    for key, session in indexed_sessions.items():
        logger.info(f"[{split_name}] at key {key}: {len(session['edf_paths'])} .edf files")
        result, file_offsets = concatenate_session_eeg(session)
        if result is None:
            logger.warning(f"[{split_name}] No .edf files for {key}")
        else:
            logger.info(f"[{split_name}] Shape: {result.shape}")
            mean = result.mean(axis=1, keepdims=True)
            std = result.std(axis=1, keepdims=True)
            std[std == 0] = 1.0  # guard against flat/dead channels
            normalized = (result - mean) / std
            normalized_sessions[key] = (normalized, file_offsets)
            logger.info(f"[{split_name}] {key} z-score normalized (per channel)")

        metadata = extract_session_metadata(session)
        logger.info(f"[{split_name}] Metadata for {key}: {metadata}")

    preictal_labels = [l for l in master_df["label"].unique()
                        if l.startswith("p") and not l.endswith("_sopbuffer")]
    preictal_windows = extract_segments(
        master_df, indexed_sessions, normalized_sessions,
        label_filter=preictal_labels
    )
    if len(preictal_windows) == 0:
        logger.error(f"[{split_name}] No preictal windows extracted — skipping segmentation for this split")
        return
    interictal_windows = extract_segments(
        master_df, indexed_sessions, normalized_sessions,
        label_filter=["interictal"]
    )
    if len(interictal_windows) == 0:
        logger.error(f"[{split_name}] No interictal windows extracted — skipping segmentation for this split")
        return

    preictal_windows = montage_windows(preictal_windows)
    interictal_windows = montage_windows(interictal_windows)

    logger.info(f"preictal windows {len(preictal_windows)}, interictal windows {len(interictal_windows)}")
    interictal_segments = segment_fixed(interictal_windows, SEG_TIME, SFREQ)

    total_len_inter = sum(w["window"].shape[1] for w in interictal_windows)
    preictal_segments = segment_adaptive(
        preictal_windows, SEG_TIME, SFREQ, total_len_inter=total_len_inter
    )

    preictal_dir = Path(f"preictal_{split_name}")
    interictal_dir = Path(f"interictal_{split_name}")

    # deletes the existing directories if they exist so old and new segments
    # dont get mixed up if running pipeline multiple times on the same split
    if preictal_dir.exists():
        shutil.rmtree(preictal_dir)
    if interictal_dir.exists():
        shutil.rmtree(interictal_dir)
    preictal_dir.mkdir(exist_ok=True)
    interictal_dir.mkdir(exist_ok=True)

    for i, s in enumerate(preictal_segments):
        np.save(preictal_dir / f"{s['label']}_{i}.npy", s["segment"])
    for i, s in enumerate(interictal_segments):
        np.save(interictal_dir / f"{s['label']}_{i}.npy", s["segment"])

    elapsed = time.time() - start_time
    logger.info(f"Preprocessing Pipeline Time {elapsed}")
    logger.info(f"[{split_name}] Saved {len(preictal_segments)} preictal, "
                f"{len(interictal_segments)} interictal segments")
    
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
def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            # label convention: 1 = preictal (positive), 0 = interictal (negative)
            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    avg_loss = total_loss / total
    acc = correct / total
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall on preictal
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall on interictal

    return avg_loss, acc, sensitivity, specificity

if __name__ == "__main__":
    run_pipeline_for_split("train", master_csv="master_full.csv")
    run_pipeline_for_split("dev", master_csv="master_dev.csv")
    
    train_dataset = EEGSegmentDataset("preictal_train", "interictal_train")
    dev_dataset = EEGSegmentDataset("preictal_dev", "interictal_dev")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    logger.info(f"Train segments: {len(train_dataset)}, Dev segments: {len(dev_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneDCNN(input_channels=20).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_dev_acc = 0
    for epoch in range(30):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        dev_loss, dev_acc, dev_sens, dev_spec = evaluate(model, dev_loader, device, criterion)
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss/train_total:.4f} train_acc={train_correct/train_total:.4f} "
    f"| dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} dev_sens={dev_sens:.4f} dev_spec={dev_spec:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), "best_model.pt")

    logger.info(f"Best dev accuracy: {best_dev_acc:.4f}")