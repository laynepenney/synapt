"""`synapt resume` must disclose which code answered it.

A version number alone does not pin which code ran (recall#952): an
editable install can keep reporting the same declared version while the
linked worktree underneath it is silently repointed -- a real recurring
production incident, not a hypothetical.
This is a call-site test -- the pure composition (`_with_provenance`) is
already covered in `test_server_provenance.py`; this proves `cmd_resume`
actually reaches it, with every unrelated collaborator mocked out the same
way `test_hook_session_start_bounded.py::_run_hook` isolates `cmd_hook`.
"""

from __future__ import annotations

import argparse
import io
import sys
from types import SimpleNamespace
from unittest.mock import patch

from synapt.recall import cli


def _run_resume(monkeypatch, tmp_path, *, provenance_line="synapt vTEST — running from /fixture (editable install)"):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "recall.db").write_bytes(b"placeholder")

    fake_index = SimpleNamespace(_session_order=["s1"], _db=None)
    fake_view = SimpleNamespace(refresh_label=None)

    out = io.StringIO()
    with patch.object(cli, "_resolve_index_dir", return_value=index_dir), \
         patch("synapt.recall.journal._journal_path", return_value=tmp_path / "journal.jsonl"), \
         patch("synapt.recall.resume.caller_transcripts", return_value=["fake-caller"]), \
         patch("synapt.recall.resume.load_resume_index", return_value=fake_index), \
         patch("synapt.recall.resume.build_resume_view", return_value=fake_view), \
         patch.object(cli, "_attach_freshness", return_value=fake_view), \
         patch.object(cli, "_attach_unclean_end", return_value=fake_view), \
         patch("synapt.recall.resume.attach_durable_checkpoint", return_value=fake_view), \
         patch("synapt.recall.resume.format_resume", return_value="RESUME BODY"), \
         patch("synapt.recall.server._query_freshness_line", return_value=""), \
         patch("synapt.recall.server._resolved_provenance_line", return_value=provenance_line), \
         patch.object(sys, "stdout", out):
        cli.cmd_resume(argparse.Namespace(session=None, turns=10, index=None, project=None))
    return out.getvalue()


def test_resume_output_carries_the_provenance_line(monkeypatch, tmp_path):
    out = _run_resume(monkeypatch, tmp_path)
    assert "RESUME BODY" in out
    assert "Provenance: synapt vTEST — running from /fixture (editable install)" in out


def test_resume_provenance_line_reflects_a_different_resolved_location(monkeypatch, tmp_path):
    """Control: change only the mocked location and the printed line changes
    with it -- proves the test isn't passing on a hardcoded string match
    that would pass regardless of what cmd_resume actually composed."""
    out = _run_resume(
        monkeypatch, tmp_path,
        provenance_line="synapt vTEST — running from /a/DIFFERENT/path (non-editable install)",
    )
    assert "Provenance: synapt vTEST — running from /a/DIFFERENT/path (non-editable install)" in out
    assert "/fixture" not in out
