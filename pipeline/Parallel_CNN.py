from util import handle_logs

from pipeline.session_index import index_sessions

logger = handle_logs.get_logger("Parallel_CNN", "logs/app.log")

if __name__ == "__main__":
    session = index_sessions("train")
