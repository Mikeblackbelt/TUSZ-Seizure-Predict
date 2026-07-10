import os, sys, types
os.environ.setdefault("MPLBACKEND", "Agg")  # headless: avoid plt.show() blocking

import numpy as np
import scipy.io as sio
from scipy.signal import firwin, filtfilt
from pathlib import Path

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
FS = 256  # EEGDenoiseNet sampling rate

# ---- Step 4: bandpass filter 1-80 Hz (replicates helpers/epochs_filter.py) ----
def custom_bandpass_filter(data, lowcut=1, highcut=80, fs=FS,
                           filter_length=101, pad_length=100):
    nyq = 0.5 * fs
    taps = firwin(filter_length, [lowcut/nyq, highcut/nyq], window="hann", pass_zero=False)
    padded = np.pad(data, pad_length, mode="edge")
    filtered = filtfilt(taps, 1.0, padded)
    return filtered[pad_length:-pad_length]

def filter_signal(name):
    src = DATA_DIR / f"{name}_all_epochs.npy"
    dst = DATA_DIR / f"filtered80Hz_{name}_all_epochs.mat"
    if dst.exists():
        print(f"  {dst.name} exists, skipping.")
        return
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")
    print(f"  filtering {src.name} -> {dst.name}")
    X = np.load(src)              # (n_epochs, 512)
    out = np.zeros_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        out[i, :] = custom_bandpass_filter(X[i, :])
    sio.savemat(str(dst), {f"{name}_all_epochs": out})

for nm in ("EEG", "EOG", "EMG"):
    filter_signal(nm)
print("[4/6] Filtering complete.")

# ---- Step 5: combine into splits (replicates datanoise_combiner.py) ----
# Reuse the repo's own combiner so train/val/test layout matches the trainer exactly.
from datanoise_combiner import DataNoiseCombiner
cfg = types.SimpleNamespace(
    datapath=str(DATA_DIR),
    lower_snr=-7.0,
    higher_snr=4.5,
    test_size=0.25,
    val_size=0.2,
)
# Clean any stale splits first (env_setup.sh behaviour).
for sub in ("train", "val", "test"):
    p = DATA_DIR / sub
    if p.exists():
        import shutil
        shutil.rmtree(p)
print("[5/6] Combining clean+noise into train/val/test splits...")
DataNoiseCombiner(cfg)
print("[5/6] Splits built.")

# ---- Step 6: train (replicates main.py without the bash env_setup.sh hook) ----
import argparse
from MLPTrainer import MLPTrainer

def load_config():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", type=str, default=str(DATA_DIR))
    p.add_argument("--outputpath", type=str, default="output")
    p.add_argument("--snr_db", type=float, default=None)
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--lower_snr", type=float, default=-7.0)
    p.add_argument("--higher_snr", type=float, default=4.5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--log_file", type=str, default="train_log.txt")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--no_plot", default=False, action="store_false")
    p.add_argument("--save_path", type=str, default=str(CKPT_DIR))
    p.add_argument("--mode", type=str, default="train")
    p.add_argument("--model", type=str, default="MLP")
    p.add_argument("--pca", default=False, action="store_true")
    p.add_argument("--ica", default=False, action="store_true")
    args, _ = p.parse_known_args()
    return args

print("[6/6] Training MLP with PCA (this takes a few minutes)...")
trainer = MLPTrainer(load_config())
trainer.run()
print("[6/6] Training complete.")
