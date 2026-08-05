import os
import time
import traceback

from util import handle_logs
from pipeline.session_index import index_sessions
from pipeline.raw_eeg_extraction import concatenate_session_eeg
from pipeline.bipolar_montages import create_bipolar_montages

logger = handle_logs.get_logger("Parallel_CNN", "logs/app.log")

# Configuration
CHECKPOINT_DIR = "checkpoints"
SPLIT = "train"
MAX_SESSIONS = 5  # Only process a small chunk of the dataset for testing

def test_pipeline_chunk():
    logger.info("-" * 60)
    logger.info(f"Starting Pipeline Orchestrator (Testing up to {MAX_SESSIONS} sessions)")
    logger.info("Pipeline stages: Indexing -> Extraction/Filtering -> Bipolar Montages")
    logger.info("-" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Index Sessions
    logger.info(f"Indexing '{SPLIT}' sessions...")
    try:
        sessions = index_sessions(SPLIT)
    except Exception as e:
        logger.error(f"Failed to index sessions: {e}")
        return

    session_keys = list(sessions.keys())
    
    if not session_keys:
        logger.warning(f"No sessions found for split: {SPLIT}")
        return

    # Slice the dictionary to only process a small chunk
    chunk_keys = session_keys[:MAX_SESSIONS]
    logger.info(f"Found {len(session_keys)} total sessions. Processing a chunk of {len(chunk_keys)}.")

    successful_raw = 0
    successful_proc = 0

    # 2. Main Processing Loop (Steps 1 through 7)
    for i, key in enumerate(chunk_keys):
        logger.info(f"\n[{i+1}/{len(chunk_keys)}] Processing Session: {key}")
        session_data = sessions[key]
        
        start_time = time.time()

        try:
            # --- STEPS 4a to 6: Raw Signal Ingestion, Filtering, Resampling, Concatenation ---
            logger.info("  -> Extracting and concatenating raw EEG...")
            
            # This automatically saves the '{key}_raw.npy' checkpoint via checkpoint_io
            result = concatenate_session_eeg(
                session=session_data,
                session_key=key,
                output_dir=CHECKPOINT_DIR,
                apply_filtering=True 
            )

            if result is None:
                logger.warning(f"  -> Skipped {key}: No valid EDF data extracted.")
                continue
                
            raw_array, file_offsets = result
            successful_raw += 1
            logger.info(f"  -> Raw extraction successful. Shape: {raw_array.shape}")

            # --- STEP 7: Convert to Bipolar Montage ---
            logger.info("  -> Converting to Bipolar Montage...")
            
            # This loads '{key}_raw.npy' and saves '{key}_proc.npy' via checkpoint_io
            # This _proc.npy file is what your collaborator will use for process_signal.py
            proc_array = create_bipolar_montages(
                session_key=key,
                checkpoint_dir=CHECKPOINT_DIR
            )

            if proc_array is not None:
                successful_proc += 1
                logger.info(f"  -> Bipolar conversion successful. Shape: {proc_array.shape}")
            else:
                logger.error(f"  -> Bipolar conversion returned None for {key}.")

        except Exception as e:
            logger.error(f"  -> FATAL ERROR processing {key}: {e}")
            logger.debug(traceback.format_exc())
            
        elapsed = time.time() - start_time
        logger.info(f"  -> Session {key} finished in {elapsed:.2f} seconds.")

    # 3. Summary
    logger.info("-" * 60)
    logger.info("Pipeline Chunk Testing Complete.")
    logger.info(f"Total Attempted: {len(chunk_keys)}")
    logger.info(f"Successful Raw Checkpoints: {successful_raw}")
    logger.info(f"Successful Processed (Bipolar) Checkpoints: {successful_proc}")
    logger.info("Ready for handoff to windowing/epoching.")
    logger.info("-" * 60)

if __name__ == "__main__":
    test_pipeline_chunk()