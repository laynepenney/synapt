"""recall#856: the wake read's own footer has always claimed (see render_wake's
docstring) "a footer that says how many bytes were shown, how many withheld, and
where the rest is" -- but the ITEM count it already computes (`withheld_lines`)
is discarded (used only as a boolean gate, never printed), and nothing reports
how far BACK the carried-forward items a reader can still see actually reach.

Measured on a real store (Sentinel, 2026-09-05, #dev): the latest journal entry
was 3,279 bytes against the 6,000-byte per-source cap -- nothing withheld today,
which is itself worth knowing rather than assuming. The gap is real regardless:
a big-enough entry clips silently on the ITEM axis (a byte count tells you
nothing about how many carried steps just vanished) and with no date marker
naming the oldest one still visible, "everything before date X may be gone"
is not answerable from the footer at all.

This constructs a journal block deliberately larger than _CAP_JOURNAL_LATEST so
clipping actually engages, and checks the footer reports an item count and the
oldest surviving carry-stamp -- not just aggregate bytes.
"""
from __future__ import annotations

from synapt.recall.session_start import render_wake, _CAP_JOURNAL_LATEST


def _render(lines, tmp_path):
    return render_wake(lines, source="startup", full_path=tmp_path / "latest.md")


def _oversized_journal_block(n_items: int = 90) -> str:
    """Newest-first, oldest-last -- the real carry-forward ordering (new steps
    first, carried steps appended each hop), so byte-clipping (keep the first
    N bytes) keeps the newest and drops the oldest, exactly as production does.
    """
    lines = ["Next steps:"]
    for i in range(n_items):
        day = 30 - (i % 28)
        date = f"2026-{(8 if day <= 28 else 7):02d}-{max(1, day):02d}"
        lines.append(
            f"  - step number {i:03d} with enough padding text to matter "
            f"[carried since {date}]"
        )
    return "\n".join(lines)


def test_footer_reports_item_count_and_oldest_shown_date_when_clipped(tmp_path):
    block = _oversized_journal_block()
    assert len(block.encode("utf-8")) > _CAP_JOURNAL_LATEST, \
        "fixture must actually exceed the cap or this test proves nothing"

    out = _render([block], tmp_path)

    footer = out.rsplit("---", 1)[-1]
    assert "withheld" in footer
    # The BYTE count alone ("N B withheld") is not an item count; the footer
    # must also say how many carried items were withheld, not just bytes.
    import re
    assert re.search(r"\d+\s*item", footer), (
        f"footer names a byte count but not an item count: {footer!r}"
    )
    # And it must name the OLDEST date still visible after clipping, not just
    # a byte total -- a reader needs to know how far back the visible window
    # reaches, not merely that something was cut. Pinned to the exact value
    # (not just "some YYYY-MM-DD"): the fixture's 28 distinct dates are not
    # in sorted order, and its true minimum ("2026-07-29", at index 1, well
    # inside the surviving prefix) is what an oldest()-vs-newest() mix-up
    # would get wrong while still matching a looser date-shaped pattern.
    assert "oldest shown: 2026-07-29" in footer, (
        f"footer names a different oldest-shown date than the fixture's true "
        f"minimum (2026-07-29): {footer!r}"
    )


def test_footer_omits_the_new_fields_when_nothing_was_withheld(tmp_path):
    """Control: a small block that fits inside the cap must not claim a
    withholding it did not do."""
    small = "Next steps:\n  - a single short step [carried since 2026-09-01]"
    out = _render([small], tmp_path)
    footer = out.rsplit("---", 1)[-1]
    assert "complete" in footer
    assert "withheld" not in footer
