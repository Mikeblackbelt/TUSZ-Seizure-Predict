from util import handle_logs

def test_logging(tmp_path):
    log_file = tmp_path / "test.log"

    logger = handle_logs.get_logger("test_logger", str(log_file))
    logger.info("This is a test log message.")

    assert log_file.exists()

