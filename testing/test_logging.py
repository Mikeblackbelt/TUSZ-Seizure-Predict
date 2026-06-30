import pytest
import pandas as pd
from pipeline.preictal_segment import (
    get_unique_tags,
    make_master_file,
    add_preictal_tags,
    get_split,
)
from util import handle_logs
from testing.helpers import *
logger = handle_logs.get_logger("test_pipeline", "logs/test.log")

def test_logging_creates_file(tmp_path):
    logger.info("test_logging_creates_file: start")
    log_file = tmp_path / "test.log"
    test_logger = handle_logs.get_logger("test_logger", str(log_file))
    test_logger.info("This is a test log message.")
    assert log_file.exists(), "Log file was not created"
    logger.info("test_logging_creates_file: passed")

def test_logging_writes_content(tmp_path):
    logger.info("test_logging_writes_content: start")
    log_file = tmp_path / "test.log"
    test_logger = handle_logs.get_logger("test_logger_content", str(log_file))
    test_logger.info("hello from test")
    assert "hello from test" in log_file.read_text()
    logger.info("test_logging_writes_content: passed")
