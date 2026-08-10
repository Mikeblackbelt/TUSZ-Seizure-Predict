from pipeline import pipeline_gpt2class


def test_get_sessions_to_process_starts_from_requested_session(monkeypatch):
    sessions = {"sess_a": object(), "sess_b": object(), "sess_c": object()}

    def fake_checkpoint_exists(checkpoint_dir, session_key, stage):
        if session_key == "sess_a" and stage == "raw":
            return True
        return False

    monkeypatch.setattr(pipeline_gpt2class, "_checkpoint_exists", fake_checkpoint_exists)

    ordered = list(
        pipeline_gpt2class.get_sessions_to_process(
            sessions,
            checkpoint_dir="checkpoints",
            create_montage_flag=True,
            start_from_session_key="sess_b",
        )
    )

    assert ordered == ["sess_b", "sess_c"]
