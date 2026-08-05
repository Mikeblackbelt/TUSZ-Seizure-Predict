from util import handle_logs
from pipeline.session_index import index_sessions     
from pipeline.raw_eeg_extraction import concatenate_session_eeg
from pipeline.session_metadata import extract_session_metadata
import time
import os
import pandas as pd
import numpy as np
from pathlib import Path
from pipeline.adaptive_windows import extract_windows

logger = handle_logs.get_logger("slimseiz", "applog")

if __name__ == "__main__":
    #index sessions
    indexed_sessions = index_sessions("train")

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
    preictal_windows = extract_windows(
        master_df, indexed_sessions, output_dir="raweeg_output",
        label_filter=preictal_labels, status_filter=[1],
    )

    interictal_windows = extract_windows(
        master_df, indexed_sessions, output_dir="raweeg_output",
        label_filter=["interictal"], status_filter=[2],
    )
    Path("preictal").mkdir(exist_ok=True)
    for i, w in enumerate(preictal_windows):
        np.save(f"preictal/{w['label']}_{i}.npy", w["window"])

    Path("interictal").mkdir(exist_ok=True)
    for i, w in enumerate(interictal_windows):
        np.save(f"interictal/{w['label']}_{i}.npy", w["window"])
       #associate the session metadata from the npy sessions with the channels/labels from the master csv to.... ignore? remove? 
        #csv bi paths for both
        #1 find the associated npy and csv
        #2 slice npys
    #drop non preictal
    
    #normalize
    #adaptive windows
    