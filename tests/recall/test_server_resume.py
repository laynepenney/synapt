"""recall_resume: the `synapt resume` surface exposed as an MCP tool.

Mirrors test_server_sessions.py: the tool must use the bounded resume index,
never construct the full index, and always close the DB it opened.
"""

from unittest.mock import Mock, patch


def test_recall_resume_reports_missing_index(monkeypatch, tmp_path):
    from synapt.recall import server

    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    out = server.recall_resume()
    assert out.startswith("No index found at")
    assert "synapt recall setup" in out


def test_recall_resume_empty_index_is_honest_empty_not_error(monkeypatch, tmp_path):
    from synapt.recall import server
    from synapt.recall.resume import ResumeError

    index = Mock()
    index._session_order = []
    index._db = Mock()
    (tmp_path / "recall.db").touch()
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    with (
        patch("synapt.recall.journal._journal_path", return_value=tmp_path / "journal.jsonl"),
        patch("synapt.recall.resume.load_resume_index", return_value=index) as load,
        patch("synapt.recall.resume.build_resume_view", side_effect=ResumeError("nothing")),
        patch.object(
            server,
            "_get_index",
            side_effect=AssertionError("resume constructed the full index"),
        ),
    ):
        assert server.recall_resume() == "No sessions indexed yet. Nothing to resume."

    load.assert_called_once_with(tmp_path)
    index._db.close.assert_called_once_with()


def test_recall_resume_unresolved_session_is_an_error_not_empty(monkeypatch, tmp_path):
    from synapt.recall import server
    from synapt.recall.resume import ResumeError

    index = Mock()
    index._session_order = ["abc"]
    index._db = Mock()
    (tmp_path / "recall.db").touch()
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    with (
        patch("synapt.recall.journal._journal_path", return_value=tmp_path / "journal.jsonl"),
        patch("synapt.recall.resume.load_resume_index", return_value=index),
        patch("synapt.recall.resume.build_resume_view", side_effect=ResumeError("no such session zzz")),
    ):
        out = server.recall_resume(session_id="zzz")

    assert out == "Resume failed: no such session zzz"
    index._db.close.assert_called_once_with()


def test_recall_resume_passes_session_and_turns_and_renders(monkeypatch, tmp_path):
    from synapt.recall import server

    index = Mock()
    index._session_order = ["abc"]
    index._db = Mock()
    view = Mock()
    view.turns = [object()]
    (tmp_path / "recall.db").touch()
    monkeypatch.setattr(server, "project_index_dir", lambda: tmp_path)
    with (
        patch("synapt.recall.journal._journal_path", return_value=tmp_path / "journal.jsonl"),
        patch("synapt.recall.resume.load_resume_index", return_value=index),
        patch("synapt.recall.resume.build_resume_view", return_value=view) as build,
        patch("synapt.recall.freshness.check_index_freshness", side_effect=RuntimeError("no fs")),
        patch("synapt.recall.resume.format_resume", return_value="RENDERED") as fmt,
    ):
        assert server.recall_resume(session_id="ab", turns=3) == "RENDERED"

    kwargs = build.call_args.kwargs
    assert kwargs["session_id"] == "ab"
    assert kwargs["limit"] == 3
    assert kwargs["journal_path"] == tmp_path / "journal.jsonl"
    # Freshness failure must not break the tool: the view renders unchanged.
    fmt.assert_called_once_with(view)
    index._db.close.assert_called_once_with()


def test_recall_resume_is_registered_with_directive_check():
    from synapt.recall import server

    registered = []

    class FakeMCP:
        def tool(self):
            def deco(fn):
                registered.append(getattr(fn, "__name__", repr(fn)))
                return fn
            return deco

    server.register_tools(FakeMCP())
    assert "recall_resume" in registered
    assert "recall_sessions" in registered
