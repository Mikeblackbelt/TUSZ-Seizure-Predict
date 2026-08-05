from util import handle_logs
from pipeline.session_index import index_sessions     
from pipeline.raw_eeg_extraction import concatenate_session_eeg
import time
logger = handle_logs.get_logger("slimseiz", "applog")

if __name__ == "__main__":
    #index sessions
    indexed_sessions = index_sessions("train")
    session_keys = list(indexed_sessions.keys())
    #concatenate sessions (drops channels, resamples, and saves to .npy)
    for key in session_keys:
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
    #drop non preictal
    #normalize
    #adaptive windows
    