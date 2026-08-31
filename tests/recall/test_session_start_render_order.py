"""render_wake puts an UNCLEAN END block ahead of every other source and caps it.

The block exists to change what the reader does first; if it renders after the
journal it is one more paragraph in a wake the harness previews at ~2 KB.
"""
from __future__ import annotations


from synapt.recall.session_start import render_wake, _CAP_UNCLEAN_END


def _render(lines, tmp_path):
    return render_wake(lines, project=tmp_path, source="startup",
                       full_path=tmp_path / "latest.md")


def test_unclean_end_renders_before_the_journal(tmp_path):
    journal = "Last session (2026-08-30T23:54): gate-day close\nNext steps:\n  - old plan"
    unclean = "UNCLEAN END — session 65262c2c ended without a handoff\nLast activity 12:06Z"
    out = _render([journal, unclean], tmp_path)
    assert out.index("UNCLEAN END") < out.index("Last session")
    control = _render([journal], tmp_path)
    assert "UNCLEAN END" not in control


def test_unclean_end_is_capped_and_says_so(tmp_path):
    big = "UNCLEAN END — session x\n" + "\n".join(f"tail line {i} " + "x" * 80 for i in range(200))
    out = _render([big], tmp_path)
    body = out.split("UNCLEAN END", 1)[1]
    assert len(body.encode()) < _CAP_UNCLEAN_END + 200
    assert "withheld" in out
