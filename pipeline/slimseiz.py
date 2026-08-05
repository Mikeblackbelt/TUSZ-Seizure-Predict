from util import handle_logs
from pipeline.session_index import index_sessions     
from pipeline.raw_eeg_extraction import concatenate_session_eeg
logger = handle_logs.get_logger("slimseiz", "applog")

if __name__ == "__main__":
    indexed_sessions = index_sessions("train")
    concatenated_session_eeg