"""TDD for a build-lock-honoring fix to the "No index found" message.

``recall_search`` (and ``recall_quick``, ``recall_resume``, which share the
exact same "No index found" sentence and already compute a freshness
trailer) said "No index found at <dir>. Run `synapt recall setup` first."
whenever ``_get_index()`` returned None -- even when the real cause was a
concurrent ``recall_build`` holding the build lock, in which case the
Freshness trailer computed moments earlier already carried
``reason=build_lock``. The primary sentence contradicted its own trailer:
one says "there is no index, create one," the other says "there is one, it
is locked." A caller reading only the first sentence (the normal case) is
told to do the wrong thing -- reinstall/setup instead of retry.

The fix: a single helper, ``_index_missing_message``, that checks whether
the build lock is actually held before choosing the sentence, and names the
holder (pid + since-timestamp, from the same stamp `_build_lock_busy_message`
already reads) when it is.
"""

from __future__ import annotations

from unittest.mock import patch

from synapt.recall.server import (
    _index_missing_message,
    recall_quick,
    recall_resume,
    recall_search,
)


def test_lock_free_index_dir_reports_genuinely_missing_index(tmp_path):
    index_dir = tmp_path / "index"
    message = _index_missing_message(index_dir)
    assert message == (
        f"No index found at {index_dir}. Run `synapt recall setup` first."
    )


def test_lock_held_reports_build_in_progress_not_missing(tmp_path):
    index_dir = tmp_path / "index"
    with (
        patch("synapt.recall.cli._acquire_build_lock", return_value=None),
        patch(
            "synapt.recall.cli._build_lock_busy_message",
            return_value="held by pid 4242 since 2026-09-02T18:00:00-05:00",
        ),
    ):
        message = _index_missing_message(index_dir)
    assert "No index found" not in message
    assert "in progress" in message
    assert "held by pid 4242 since 2026-09-02T18:00:00-05:00" in message
    assert str(index_dir) in message


def test_historical_lock_free_reports_genuinely_missing_index(tmp_path):
    index_dir = tmp_path / "index"
    message = _index_missing_message(index_dir, historical=True)
    assert "Historical search unavailable: no index found" in message
    assert "date-filtered query without an index" in message


def test_historical_lock_held_reports_build_in_progress_not_missing(tmp_path):
    index_dir = tmp_path / "index"
    with (
        patch("synapt.recall.cli._acquire_build_lock", return_value=None),
        patch(
            "synapt.recall.cli._build_lock_busy_message",
            return_value="held by pid 4242 since 2026-09-02T18:00:00-05:00",
        ),
    ):
        message = _index_missing_message(index_dir, historical=True)
    assert "No index found" not in message
    assert "in progress" in message
    assert "held by pid 4242 since 2026-09-02T18:00:00-05:00" in message
    assert "date-filtered query" in message


def _busy_freshness_line():
    return (
        "Freshness: BUSY session=c7dc26db indexed_through=unknown "
        "live_bytes=944 indexed_now=0B/0chunks wall=0.000s remaining=unknown "
        "cut_short=true reason=build_lock"
    )


class TestRecallSearchHonorsBuildLock:
    def test_search_message_agrees_with_its_own_freshness_trailer(self, tmp_path):
        with (
            patch("synapt.recall.server._get_index", return_value=None),
            patch("synapt.recall.server.project_index_dir", return_value=tmp_path),
            patch(
                "synapt.recall.server._query_freshness_line",
                return_value=_busy_freshness_line(),
            ),
            patch(
                "synapt.recall.source_index.search_registered_sources",
                return_value=[],
            ),
            patch("synapt.recall.live.search_live_transcript", return_value=""),
            patch("synapt.recall.cli._acquire_build_lock", return_value=None),
            patch(
                "synapt.recall.cli._build_lock_busy_message",
                return_value="held by pid 4242 since 2026-09-02T18:00:00-05:00",
            ),
        ):
            result = recall_search("checkout edge case")

        assert "reason=build_lock" in result
        assert "No index found" not in result
        assert "in progress" in result

    def test_historical_filtered_search_message_agrees_with_its_own_freshness_trailer(
        self, tmp_path
    ):
        with (
            patch("synapt.recall.server._get_index", return_value=None),
            patch("synapt.recall.server.project_index_dir", return_value=tmp_path),
            patch(
                "synapt.recall.server._query_freshness_line",
                return_value=_busy_freshness_line(),
            ),
            patch(
                "synapt.recall.source_index.search_registered_sources",
                return_value=[],
            ),
            patch("synapt.recall.live.search_live_transcript", return_value=""),
            patch("synapt.recall.cli._acquire_build_lock", return_value=None),
            patch(
                "synapt.recall.cli._build_lock_busy_message",
                return_value="held by pid 4242 since 2026-09-02T18:00:00-05:00",
            ),
        ):
            result = recall_search("checkout edge case", after="2026-08-01")

        assert "reason=build_lock" in result
        assert "No index found" not in result
        assert "in progress" in result
        assert "date-filtered query" in result


class TestRecallQuickHonorsBuildLock:
    def test_quick_message_agrees_with_its_own_freshness_trailer(self, tmp_path):
        with (
            patch("synapt.recall.server._get_index", return_value=None),
            patch("synapt.recall.server.project_index_dir", return_value=tmp_path),
            patch(
                "synapt.recall.server._query_freshness_line",
                return_value=_busy_freshness_line(),
            ),
            patch("synapt.recall.cli._acquire_build_lock", return_value=None),
            patch(
                "synapt.recall.cli._build_lock_busy_message",
                return_value="held by pid 4242 since 2026-09-02T18:00:00-05:00",
            ),
        ):
            result = recall_quick("checkout edge case")

        assert "reason=build_lock" in result
        assert "No index found" not in result
        assert "in progress" in result


class TestRecallResumeHonorsBuildLock:
    def test_resume_message_agrees_with_its_own_freshness_trailer(self, tmp_path):
        with (
            patch("synapt.recall.server.project_index_dir", return_value=tmp_path),
            patch(
                "synapt.recall.server._query_freshness_line",
                return_value=_busy_freshness_line(),
            ),
            patch("synapt.recall.cli._acquire_build_lock", return_value=None),
            patch(
                "synapt.recall.cli._build_lock_busy_message",
                return_value="held by pid 4242 since 2026-09-02T18:00:00-05:00",
            ),
        ):
            result = recall_resume()

        assert "reason=build_lock" in result
        assert "No index found" not in result
        assert "in progress" in result
