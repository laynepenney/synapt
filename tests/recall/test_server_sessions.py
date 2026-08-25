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
