import os

from pipeline_gpt2class import resolve_output_path


def test_resolve_output_path_uses_default_filename_for_directory(tmp_path):
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()

    resolved = resolve_output_path(str(output_dir))

    assert resolved == os.path.join(str(output_dir), "master_full.csv")
