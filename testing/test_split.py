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

def test_get_split_train():
    logger.info("test_get_split_train: start")
    assert get_split("/data/tusz/edf/train/01_tcp_ar/rec.edf") == "train"
    logger.info("test_get_split_train: passed")

def test_get_split_dev():
    logger.info("test_get_split_dev: start")
    assert get_split("/data/tusz/edf/dev/01_tcp_ar/rec.edf") == "dev"
    logger.info("test_get_split_dev: passed")

def test_get_split_eval():
    logger.info("test_get_split_eval: start")
    assert get_split("/data/tusz/edf/eval/01_tcp_ar/rec.edf") == "eval"
    logger.info("test_get_split_eval: passed")

def test_get_split_case_insensitive():
    logger.info("test_get_split_case_insensitive: start")
    assert get_split("/data/tusz/edf/TRAIN/01_tcp_ar/rec.edf") == "train"
    logger.info("test_get_split_case_insensitive: passed")

def test_get_split_unknown():
    logger.info("test_get_split_unknown: start")
    assert get_split("/data/tusz/edf/misc/01_tcp_ar/rec.edf") == "unknown"
    logger.info("test_get_split_unknown: passed")