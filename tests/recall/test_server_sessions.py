from unittest.mock import Mock, patch


def test_recall_sessions_uses_the_bounded_no_embedding_surface(monkeypatch, tmp_path):
    from synapt.recall import server

    index = Mock()
    index.list_sessions.return_value = []
    index._db = Mock()
    (tmp_path / "recall.db").touch()
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    with (
        patch("synapt.recall.resume.load_resume_index", return_value=index) as load,
        patch.object(
            server,
            "_get_index",
            side_effect=AssertionError("session browsing constructed the full index"),
        ),
    ):
        assert server.recall_sessions() == "No sessions found."

    load.assert_called_once_with(tmp_path)
    index._db.close.assert_called_once_with()


def test_recall_sessions_labels_each_row_with_its_source_root(monkeypatch, tmp_path):
    from synapt.recall import server

    index = Mock()
    index.list_sessions.return_value = [{
        "date": "2026-08-29",
        "session_id": "aaaaaaaa-1111",
        "turn_count": 3,
        "files_count": 1,
        "source_root": "worktree:atlas",
        "first_message": "question",
    }]
    index._db = Mock()
    (tmp_path / "recall.db").touch()
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    with patch("synapt.recall.resume.load_resume_index", return_value=index):
        out = server.recall_sessions()
    assert "[worktree:atlas]" in out
