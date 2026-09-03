"""Residue of the knowledge-store routing class, found during review:
four "no index found" existence checks -- ``cmd_search``, ``cmd_benchmark``,
``cmd_stats`` (CLI) and ``recall_stats`` (MCP tool) -- tested only
``recall.db`` or a legacy sibling file (``chunks.jsonl`` / ``manifest.json``),
never ``is_sharded(index_dir)``. On a genuinely sharded store (only
``index.db`` present -- the layout the routing fix just made MORE common by
correctly writing new content there), every one of these four falsely
reported "no index found" even though a real, valid index exists.

Two sibling commands in the same file already had this right
(``cmd_sessions`` at cli.py, ``recall_sessions``/``recall_resume`` at
server.py all include ``and not is_sharded(index_dir)`` in the same
compound condition) -- the fix here is bringing these four in line with
that already-correct pattern, not inventing a new one.

Each site gets a sharded-store test (empty ``index.db``, no ``recall.db``,
no legacy sibling file -- the exact shape a store gets when it is sharded
from birth or has had its stale ``recall.db`` cleaned up) plus a
monolithic control confirming the untouched, no-``index.db`` case still
behaves exactly as before.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.storage import RecallDB


def _sharded_dir() -> Path:
    d = Path(tempfile.mkdtemp())
    RecallDB(d / "index.db").close()  # real schema, the sharded marker
    return d


def _monolithic_dir() -> Path:
    d = Path(tempfile.mkdtemp())  # no index.db at all
    RecallDB(d / "recall.db").close()
    return d


def _empty_dir() -> Path:
    return Path(tempfile.mkdtemp())  # genuinely no index anywhere


class TestCmdSearchRouting(unittest.TestCase):

    def _run(self, index_dir: Path) -> str:
        from synapt.recall.cli import cmd_search

        args = argparse.Namespace(
            index=str(index_dir), out=None, profile=False,
            query="anything", max_chunks=5, max_tokens=None,
            max_sessions=None, after=None, before=None,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                cmd_search(args)
            except SystemExit:
                pass
        return buf.getvalue()

    def test_sharded_store_is_not_reported_missing(self):
        output = self._run(_sharded_dir())
        self.assertNotIn("no index found", output.lower())

    def test_monolithic_store_control_unaffected(self):
        output = self._run(_monolithic_dir())
        self.assertNotIn("no index found", output.lower())

    def test_genuinely_empty_dir_still_reports_missing(self):
        output = self._run(_empty_dir())
        self.assertIn("no index found", output.lower())


class TestCmdBenchmarkRouting(unittest.TestCase):

    def _run(self, index_dir: Path) -> str:
        from synapt.recall.cli import cmd_benchmark

        args = argparse.Namespace(
            index=str(index_dir), json_output=False, queries=None, iterations=1,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                cmd_benchmark(args)
            except SystemExit:
                pass
        return buf.getvalue()

    def test_sharded_store_is_not_reported_missing(self):
        output = self._run(_sharded_dir())
        self.assertNotIn("no index found", output.lower())

    def test_monolithic_store_control_unaffected(self):
        output = self._run(_monolithic_dir())
        self.assertNotIn("no index found", output.lower())

    def test_genuinely_empty_dir_still_reports_missing(self):
        output = self._run(_empty_dir())
        self.assertIn("no index found", output.lower())


class TestCmdStatsRouting(unittest.TestCase):

    def _run(self, index_dir: Path) -> str:
        # cmd_stats's "Archived:" line reads project_archive_dir() off cwd
        # regardless of --index (a pre-existing quirk, unrelated to this
        # fix -- separately worth a follow-up), so cwd must sit under the
        # same pytest-owned tmp root as index_dir for the isolation guard.
        import os

        from synapt.recall.cli import cmd_stats

        args = argparse.Namespace(index=str(index_dir), out=None)
        buf = io.StringIO()
        cwd_before = os.getcwd()
        os.chdir(index_dir)
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    cmd_stats(args)
                except SystemExit:
                    pass
        finally:
            os.chdir(cwd_before)
        return buf.getvalue()

    def test_sharded_store_is_not_reported_missing(self):
        output = self._run(_sharded_dir())
        self.assertNotIn("no index found", output.lower())

    def test_monolithic_store_control_unaffected(self):
        output = self._run(_monolithic_dir())
        self.assertNotIn("no index found", output.lower())

    def test_genuinely_empty_dir_still_reports_missing(self):
        output = self._run(_empty_dir())
        self.assertIn("no index found", output.lower())


class TestRecallStatsMcpToolRouting(unittest.TestCase):

    def _run(self, index_dir: Path) -> str:
        from synapt.recall import server as server_mod

        server_mod._cached_index = None
        server_mod._cached_dir = None
        server_mod._cached_mtime = ()
        with patch.object(server_mod, "project_index_dir", return_value=index_dir):
            return server_mod.recall_stats()

    def test_sharded_store_is_not_reported_missing(self):
        output = self._run(_sharded_dir())
        self.assertNotIn("no index found", output.lower())

    def test_monolithic_store_control_unaffected(self):
        output = self._run(_monolithic_dir())
        self.assertNotIn("no index found", output.lower())

    def test_genuinely_empty_dir_still_reports_missing(self):
        output = self._run(_empty_dir())
        self.assertIn("no index found", output.lower())


if __name__ == "__main__":
    unittest.main()
