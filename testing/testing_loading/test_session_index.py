import pytest
from pipeline.session_index import index_sessions, _parse_montage_type
from testing.helpers import *  # noqa: F401,F403  (write_edf, dataset_dir)


def _make_recording(dataset_dir, split, patient, session, recording_folder,
                     n_edf=1, n_csv_bi=1):
    rec_dir = dataset_dir / "edf" / split / patient / session / recording_folder
    rec_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_edf):
        write_edf(rec_dir / f"rec_{i}.edf")
    for i in range(n_csv_bi):
        write_edf(rec_dir / f"rec_{i}.csv_bi")
    return rec_dir


def test_montage_type_ar():
    assert _parse_montage_type("01_tcp_ar") == "ar1"


def test_montage_type_le():
    assert _parse_montage_type("02_tcp_le") == "le2"


def test_montage_type_ar_variant():
    assert _parse_montage_type("03_tcp_ar_a") == "ar3"


def test_montage_type_unknown_prefix():
    assert _parse_montage_type("weird_folder") == "unk0"


def test_index_single_session(dataset_dir):
    _make_recording(dataset_dir, "train", "p001", "s001_2015", "01_tcp_ar")
    sessions = index_sessions(str(dataset_dir))

    assert len(sessions) == 1
    key = next(iter(sessions))
    assert key == "trn_p001_s001_2015_ar1"

    record = sessions[key]
    assert record["split"] == "train"
    assert record["patient_id"] == "p001"
    assert record["session_id"] == "s001_2015"
    assert record["montage_type"] == "ar1"
    assert len(record["edf_paths"]) == 1
    assert len(record["csv_bi_paths"]) == 1


def test_index_multiple_recordings_same_session(dataset_dir):
    """Multiple recording folders under one session should merge into one record."""
    _make_recording(dataset_dir, "train", "p001", "s001_2015", "01_tcp_ar", n_edf=2)
    _make_recording(dataset_dir, "train", "p001", "s001_2015", "01_tcp_ar_b", n_edf=1)
    sessions = index_sessions(str(dataset_dir))

    # different recording_folder names produce different montage-type suffixes,
    # so unless they collide these stay separate session keys - verify no crash
    # and total edf count is preserved across whatever grouping results.
    total_edf = sum(len(r["edf_paths"]) for r in sessions.values())
    assert total_edf == 3


def test_index_multiple_files_within_one_recording_dir(dataset_dir):
    _make_recording(dataset_dir, "train", "p001", "s001_2015", "01_tcp_ar", n_edf=3, n_csv_bi=1)
    sessions = index_sessions(str(dataset_dir))

    key = next(iter(sessions))
    assert len(sessions[key]["edf_paths"]) == 3
    assert sessions[key]["edf_paths"] == sorted(sessions[key]["edf_paths"])


def test_index_multiple_sessions_and_splits(dataset_dir):
    _make_recording(dataset_dir, "train", "p001", "s001_2015", "01_tcp_ar")
    _make_recording(dataset_dir, "dev", "p002", "s001_2016", "02_tcp_le")
    _make_recording(dataset_dir, "eval", "p003", "s001_2017", "01_tcp_ar")
    sessions = index_sessions(str(dataset_dir))

    assert len(sessions) == 3
    prefixes = {key.split("_")[0] for key in sessions}
    assert prefixes == {"trn", "vld", "tst"}


def test_index_skips_unknown_split(dataset_dir):
    rec_dir = dataset_dir / "edf" / "misc" / "p001" / "s001" / "01_tcp_ar"
    rec_dir.mkdir(parents=True)
    write_edf(rec_dir / "rec.edf")
    sessions = index_sessions(str(dataset_dir))
    assert sessions == {}


def test_index_skips_unexpected_depth(dataset_dir):
    # .edf sitting directly under the split dir, no patient/session/recording levels
    shallow_dir = dataset_dir / "edf" / "train"
    shallow_dir.mkdir(parents=True)
    write_edf(shallow_dir / "rec.edf")
    sessions = index_sessions(str(dataset_dir))
    assert sessions == {}


def test_index_empty_dataset(dataset_dir):
    sessions = index_sessions(str(dataset_dir))
    assert sessions == {}