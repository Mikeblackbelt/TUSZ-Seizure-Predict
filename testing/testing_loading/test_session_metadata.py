import pytest
from pipeline.session_metadata import extract_session_metadata, _parse_duration


def _write_csv_bi(path, duration, rows):
    """
    Write a fake .csv_bi file matching TUSZ's real header/column layout.
    rows: list of (start_time, end_time, label) tuples.
    """
    lines = [
        "# version = tse_v1.0.0",
        "# bname = fake_recording",
        f"# duration = {duration:.4f} secs",
        "# montage_file = fake_montage.txt",
        "#",
        "channel,start_time,stop_time,label,confidence",
    ]
    for start, end, label in rows:
        lines.append(f"TERM,{start:.4f},{end:.4f},{label},1.0000")
    path.write_text("\n".join(lines) + "\n")


def test_parse_duration_basic(tmp_path):
    lines = [
        "# version = tse_v1.0.0",
        "# bname = fake",
        "# duration = 301.0000 secs",
    ]
    assert _parse_duration(lines, str(tmp_path)) == 301.0


def test_parse_duration_missing_header():
    assert _parse_duration(["only", "two"], "fake.csv_bi") is None


def test_single_recording_single_seizure(tmp_path):
    f = tmp_path / "rec1.csv_bi"
    _write_csv_bi(f, duration=301.0, rows=[
        (0.0, 50.0, "bckg"),
        (50.0, 80.0, "seiz"),
        (80.0, 301.0, "bckg"),
    ])
    session = {"csv_bi_paths": [str(f)]}
    result = extract_session_metadata(session)

    assert len(result["seizures"]) == 1
    seizure = result["seizures"][0]
    assert seizure["start_time"] == 50.0
    assert seizure["end_time"] == 80.0
    assert seizure["cumulative_start_time"] == 50.0
    assert seizure["cumulative_end_time"] == 80.0
    assert result["total_duration"] == 301.0


def test_multiple_recordings_offsets_second_seizure(tmp_path):
    """Seizure in the 2nd recording should be shifted by the 1st recording's duration."""
    f1 = tmp_path / "rec1.csv_bi"
    f2 = tmp_path / "rec2.csv_bi"
    _write_csv_bi(f1, duration=300.0, rows=[(0.0, 300.0, "bckg")])
    _write_csv_bi(f2, duration=200.0, rows=[(10.0, 40.0, "seiz")])

    session = {"csv_bi_paths": [str(f1), str(f2)]}
    result = extract_session_metadata(session)

    assert len(result["seizures"]) == 1
    seizure = result["seizures"][0]
    assert seizure["start_time"] == 10.0
    assert seizure["cumulative_start_time"] == 310.0  # 300 (rec1) + 10
    assert seizure["cumulative_end_time"] == 340.0     # 300 (rec1) + 40
    assert result["total_duration"] == 500.0


def test_no_seizures(tmp_path):
    f = tmp_path / "rec1.csv_bi"
    _write_csv_bi(f, duration=100.0, rows=[(0.0, 100.0, "bckg")])
    session = {"csv_bi_paths": [str(f)]}
    result = extract_session_metadata(session)

    assert result["seizures"] == []
    assert result["total_duration"] == 100.0
    assert len(result["recordings"]) == 1


def test_empty_session():
    result = extract_session_metadata({"csv_bi_paths": []})
    assert result == {"recordings": [], "seizures": [], "total_duration": 0.0}


def test_missing_csv_bi_paths_key():
    """session.get(...) default should prevent a KeyError if the key is absent."""
    result = extract_session_metadata({})
    assert result["seizures"] == []
    assert result["total_duration"] == 0.0


def test_multiple_seizures_in_one_recording(tmp_path):
    f = tmp_path / "rec1.csv_bi"
    _write_csv_bi(f, duration=200.0, rows=[
        (0.0, 20.0, "bckg"),
        (20.0, 40.0, "seiz"),
        (40.0, 60.0, "bckg"),
        (60.0, 90.0, "seiz"),
        (90.0, 200.0, "bckg"),
    ])
    session = {"csv_bi_paths": [str(f)]}
    result = extract_session_metadata(session)

    assert len(result["seizures"]) == 2
    assert [s["start_time"] for s in result["seizures"]] == [20.0, 60.0]


def test_malformed_duration_header_skips_recording(tmp_path):
    f = tmp_path / "rec1.csv_bi"
    lines = [
        "# version = tse_v1.0.0",
        "# bname = fake",
        "# duration = not_a_number secs",
        "# montage_file = fake_montage.txt",
        "#",
        "channel,start_time,stop_time,label,confidence",
        "TERM,0.0000,50.0000,seiz,1.0000",
    ]
    f.write_text("\n".join(lines))
    session = {"csv_bi_paths": [str(f)]}
    result = extract_session_metadata(session)

    assert result["recordings"] == []
    assert result["seizures"] == []
    assert result["total_duration"] == 0.0


def test_malformed_row_skipped_but_others_processed(tmp_path):
    f = tmp_path / "rec1.csv_bi"
    lines = [
        "# version = tse_v1.0.0",
        "# bname = fake",
        "# duration = 100.0000 secs",
        "# montage_file = fake_montage.txt",
        "#",
        "channel,start_time,stop_time,label,confidence",
        "TERM,bad,50.0000,seiz,1.0000",
        "TERM,60.0000,90.0000,seiz,1.0000",
    ]
    f.write_text("\n".join(lines))
    session = {"csv_bi_paths": [str(f)]}
    result = extract_session_metadata(session)

    assert len(result["seizures"]) == 1
    assert result["seizures"][0]["start_time"] == 60.0