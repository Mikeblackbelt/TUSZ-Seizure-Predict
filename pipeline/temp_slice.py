from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.session_metadata import index_sessions
from pipeline.slice_npy import extract_windows

available_keys = {p.stem.removesuffix("_raw") for p in Path("raweeg_output").glob("*_raw.npy")}
print(f"{len(available_keys)} sessions available")

sessions = index_sessions("train")
subset = {k: v for k, v in sessions.items() if k in available_keys}

master_df = pd.read_csv("master_full.csv")

preictal_labels = [l for l in master_df["label"].unique()
                    if l.startswith("p") and not l.endswith("_sopbuffer")]
preictal_windows = extract_windows(
    master_df, subset, output_dir="raweeg_output",
    label_filter=preictal_labels, status_filter=[1],
)
interictal_windows = extract_windows(
    master_df, subset, output_dir="raweeg_output",
    label_filter=["interictal"], status_filter=[2],
)

print(f"{len(preictal_windows)} preictal, {len(interictal_windows)} interictal")

Path("preictal").mkdir(exist_ok=True)
for i, w in enumerate(preictal_windows):
    np.save(f"preictal/{w['label']}_{i}.npy", w["window"])

Path("interictal").mkdir(exist_ok=True)
for i, w in enumerate(interictal_windows):
    np.save(f"interictal/{w['label']}_{i}.npy", w["window"])