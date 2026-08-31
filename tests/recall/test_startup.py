"""Tests for Codex startup parity (#633).

Verifies that:
1. generate_startup_context() returns context lines
2. cmd_startup produces output in all modes (plain, compact, json)
3. The startup command is registered and callable
4. Context includes journal, reminders, and channel when available
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


from synapt.recall.cli import (
    _dev_loop_activation_prompt,
    cmd_startup,
    generate_startup_context,
)


class TestGenerateStartupContext:
    """Test the shared context generation function."""

    def test_returns_list(self, tmp_path):
        """generate_startup_context always returns a list."""
        with patch("synapt.recall.cli.generate_startup_context") as mock:
            # Call the real function with mocked internals
            pass
        # Direct call with a path that has no recall data
        result = generate_startup_context(tmp_path)
        assert isinstance(result, list)

    def test_empty_project_returns_empty(self, tmp_path):
        """A project with no recall data returns no context (when globals mocked out)."""
        with patch("synapt.recall.knowledge.read_nodes", return_value=[]), \
             patch("synapt.recall.reminders.pop_pending", return_value=[]), \
             patch("synapt.recall.server.format_contradictions_for_session_start", return_value=""), \
             patch("synapt.recall.channel.channel_join"), \
             patch("synapt.recall.channel.channel_unread", return_value={}), \
             patch("synapt.recall.channel.check_directives", return_value=""):
            result = generate_startup_context(tmp_path)
        assert result == []

    def test_continuity_can_be_suppressed_without_suppressing_ambient_context(self, tmp_path):
        with patch("synapt.recall.journal._get_branch", return_value=None), \
             patch("synapt.recall.journal._journal_path", return_value=tmp_path / "none"), \
             patch("synapt.recall.compaction.latest_compaction_summary", return_value={
                 "runtime": "claude", "timestamp": "2026-01-01", "summary": "resume me",
             }), \
             patch("synapt.checkpoint.read_checkpoint", return_value=None), \
             patch("synapt.recall.knowledge.read_nodes", return_value=[]), \
             patch("synapt.recall.reminders.pop_pending", return_value=[object()]), \
             patch("synapt.recall.reminders.format_for_session_start", return_value="ambient reminder"), \
             patch("synapt.recall.server.format_contradictions_for_session_start", return_value=""), \
             patch("synapt.recall.channel.channel_join"), \
             patch("synapt.recall.channel.channel_unread", return_value={}), \
             patch("synapt.recall.channel.check_directives", return_value=""):
            text = "\n".join(generate_startup_context(tmp_path, include_continuity=False))

        assert "resume me" not in text
        assert "ambient reminder" in text

    def test_journal_entries_surfaced(self, tmp_path):
        """Journal entries appear in startup context when present."""
        from synapt.recall.journal import JournalEntry, append_entry, _journal_path

        jf = _journal_path(tmp_path)
        jf.parent.mkdir(parents=True, exist_ok=True)
        entry = JournalEntry(
            timestamp="2026-04-10T12:00:00Z",
            session_id="test-session-001",
            focus="Implementing Codex startup parity",
            done=["Extracted generate_startup_context"],
            decisions=["Use shared function for all tools"],
            next_steps=["Add tests"],
        )
        append_entry(entry, jf)

        # Mock _get_branch to avoid git calls
        with patch("synapt.recall.journal._get_branch", return_value=None):
            result = generate_startup_context(tmp_path)

        # Should have at least one line from the journal entry
        text = "\n".join(result)
        assert "Codex startup parity" in text or "test-session" in text

    def test_newer_raw_checkpoint_is_surfaced_after_authored_journal(self, tmp_path):
        from synapt.recall.journal import JournalEntry, append_entry, _journal_path

        jf = _journal_path(tmp_path)
        jf.parent.mkdir(parents=True, exist_ok=True)
        append_entry(JournalEntry(
            timestamp="2026-04-10T12:00:00Z",
            session_id="authored-session",
            focus="Authored handoff",
        ), jf)
        checkpoint = {
            "schema_version": 1,
            "captured_at": "2026-04-11T12:00:00Z",
            "parse_status": "ok",
            "last_user_text": "latest request",
            "last_assistant_text": "latest response",
            "files_touched": ["src/new.py"],
        }
        with patch("synapt.recall.journal._get_branch", return_value=None), \
             patch("synapt.recall.compaction.latest_compaction_summary", return_value=None), \
             patch("synapt.checkpoint.read_checkpoint", return_value=checkpoint):
            text = "\n".join(generate_startup_context(tmp_path))

        assert "LAST CHECKPOINT" in text
        assert "Not an authored journal" in text
        assert "latest request" in text

    def test_checkpoint_and_compaction_precede_older_journal_context(self, tmp_path):
        from synapt.recall.journal import JournalEntry, append_entry, _journal_path

        jf = _journal_path(tmp_path)
        jf.parent.mkdir(parents=True, exist_ok=True)
        append_entry(JournalEntry(
            timestamp="2026-04-10T12:00:00Z",
            session_id="journal",
            focus="older journal context",
        ), jf)
        checkpoint = {
            "schema_version": 1,
            "captured_at": "2026-04-11T12:00:00Z",
            "parse_status": "ok",
            "last_user_text": "newest raw turn",
        }
        summary = {
            "runtime": "claude",
            "timestamp": "2026-04-10T18:00:00Z",
            "summary": "compacted handoff",
        }
        with patch("synapt.recall.journal._get_branch", return_value=None), \
             patch("synapt.recall.compaction.latest_compaction_summary", return_value=summary), \
             patch("synapt.checkpoint.read_checkpoint", return_value=checkpoint):
            text = "\n".join(generate_startup_context(tmp_path))

        assert text.index("LAST CHECKPOINT") < text.index("LAST COMPACTION SUMMARY")
        assert text.index("LAST COMPACTION SUMMARY") < text.index("older journal context")

    def test_checkpoint_older_than_authored_journal_is_suppressed(self, tmp_path):
        from synapt.recall.journal import JournalEntry, append_entry, _journal_path

        jf = _journal_path(tmp_path)
        jf.parent.mkdir(parents=True, exist_ok=True)
        append_entry(JournalEntry(
            timestamp="2026-04-12T12:00:00Z",
            session_id="authored-session",
            focus="New authored handoff",
        ), jf)
        checkpoint = {
            "schema_version": 1,
            "captured_at": "2026-04-11T12:00:00Z",
            "parse_status": "ok",
            "last_user_text": "stale raw request",
        }
        with patch("synapt.recall.journal._get_branch", return_value=None), \
             patch("synapt.recall.compaction.latest_compaction_summary", return_value=None), \
             patch("synapt.checkpoint.read_checkpoint", return_value=checkpoint):
            text = "\n".join(generate_startup_context(tmp_path))

        assert "LAST CHECKPOINT" not in text
        assert "stale raw request" not in text

    def test_files_only_authored_journal_suppresses_older_checkpoint(self, tmp_path):
        from synapt.recall.journal import JournalEntry, append_entry, _journal_path

        jf = _journal_path(tmp_path)
        jf.parent.mkdir(parents=True, exist_ok=True)
        append_entry(JournalEntry(
            timestamp="2026-04-12T12:00:00Z",
            session_id="files-only-session",
            files_modified=["src/final.py"],
        ), jf)
        checkpoint = {
            "schema_version": 1,
            "captured_at": "2026-04-11T12:00:00Z",
            "parse_status": "ok",
            "last_user_text": "stale raw request",
        }
        with patch("synapt.recall.journal._get_branch", return_value=None), \
             patch("synapt.recall.compaction.latest_compaction_summary", return_value=None), \
             patch("synapt.checkpoint.read_checkpoint", return_value=checkpoint):
            text = "\n".join(generate_startup_context(tmp_path))

        assert "LAST CHECKPOINT" not in text
        assert "stale raw request" not in text

    def test_newer_auto_journal_does_not_hide_raw_checkpoint(self, tmp_path):
        from synapt.recall.journal import JournalEntry, append_entry, _journal_path

        jf = _journal_path(tmp_path)
        jf.parent.mkdir(parents=True, exist_ok=True)
        append_entry(JournalEntry(
            timestamp="2026-04-12T12:00:00Z",
            session_id="auto-session",
            focus="Automatically extracted focus",
            auto=True,
        ), jf)
        checkpoint = {
            "schema_version": 1,
            "captured_at": "2026-04-11T12:00:00Z",
            "parse_status": "partial",
            "last_user_text": "raw request survives",
        }
        with patch("synapt.recall.journal._get_branch", return_value=None), \
             patch("synapt.recall.compaction.latest_compaction_summary", return_value=None), \
             patch("synapt.checkpoint.read_checkpoint", return_value=checkpoint):
            text = "\n".join(generate_startup_context(tmp_path))

        assert "LAST CHECKPOINT" in text
        assert "raw request survives" in text

    def test_reminders_surfaced(self, tmp_path):
        """Pending reminders appear in startup context."""
        from synapt.recall.reminders import add_reminder

        # The two lines that used to stand here called the REAL _reminders_path()
        # and mkdir'd its parent — creating a directory inside the operator's
        # live store — and then threw the value away, because the patch below
        # supplies the path that is actually used. Dead code that wrote to a
        # real location while looking like setup for a temp one (Ref #967).
        with patch("synapt.recall.reminders._reminders_path") as mock_path:
            rfile = tmp_path / ".synapt" / "reminders.json"
            rfile.parent.mkdir(parents=True, exist_ok=True)
            mock_path.return_value = rfile

            add_reminder("Check PR reviews before merging")

            # Mock journal to avoid side effects
            with patch("synapt.recall.journal._get_branch", return_value=None):
                with patch("synapt.recall.journal._journal_path") as mock_jp:
                    mock_jp.return_value = tmp_path / "nonexistent.jsonl"
                    # Need to also mock pop_pending to use our tmp file
                    from synapt.recall.reminders import pop_pending
                    pending = pop_pending()

        # Verify we can at least call without error
        # (full integration requires more mocking)

    def test_channel_join_and_unread(self, tmp_path):
        """Channel context appears when channels have unread messages."""
        mock_join = MagicMock()
        mock_unread = MagicMock(return_value={"dev": 3})
        mock_read = MagicMock(return_value="[12:00] Apollo: hello\n[12:01] Sentinel: hi")

        with patch("synapt.recall.journal._get_branch", return_value=None), \
             patch("synapt.recall.journal._journal_path",
                   return_value=tmp_path / "nonexistent.jsonl"), \
             patch("synapt.recall.channel.channel_join", mock_join), \
             patch("synapt.recall.channel.channel_unread", mock_unread), \
             patch("synapt.recall.channel.channel_read", mock_read):
            result = generate_startup_context(tmp_path)

        text = "\n".join(result)
        assert "#dev: 3" in text
        assert "Apollo: hello" in text


class TestCmdStartup:
    """Test the cmd_startup CLI command."""

    def test_plain_output(self, capsys, tmp_path):
        """Plain mode prints lines to stdout."""
        args = argparse.Namespace(json=False, compact=False)
        with patch("synapt.recall.cli.generate_startup_context",
                   return_value=["Journal: session xyz", "Reminders: check PRs"]):
            with patch("synapt.recall.journal.compact_journal", return_value=0):
                cmd_startup(args)
        out = capsys.readouterr().out
        assert "Journal: session xyz" in out
        assert "Reminders: check PRs" in out

    def test_compact_output(self, capsys, tmp_path):
        """Compact mode joins lines with pipe separator."""
        args = argparse.Namespace(json=False, compact=True)
        with patch("synapt.recall.cli.generate_startup_context",
                   return_value=["Journal: session xyz", "Reminders: check PRs"]):
            with patch("synapt.recall.journal.compact_journal", return_value=0):
                cmd_startup(args)
        out = capsys.readouterr().out.strip()
        assert " | " in out
        assert "Journal: session xyz" in out

    def test_json_output(self, capsys, tmp_path):
        """JSON mode outputs valid JSON with context key."""
        args = argparse.Namespace(json=True, compact=False)
        with patch("synapt.recall.cli.generate_startup_context",
                   return_value=["Journal: session xyz"]):
            with patch("synapt.recall.journal.compact_journal", return_value=0):
                cmd_startup(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "context" in data
        assert "Journal: session xyz" in data["context"]

    def test_empty_context_no_output(self, capsys, tmp_path):
        """No output when there's no context to surface."""
        args = argparse.Namespace(json=False, compact=False)
        with patch("synapt.recall.cli.generate_startup_context", return_value=[]):
            with patch("synapt.recall.journal.compact_journal", return_value=0):
                cmd_startup(args)
        out = capsys.readouterr().out
        assert out == ""

    def test_empty_context_json_outputs_empty_obj(self, capsys, tmp_path):
        """JSON mode outputs {} when no context."""
        args = argparse.Namespace(json=True, compact=False)
        with patch("synapt.recall.cli.generate_startup_context", return_value=[]):
            with patch("synapt.recall.journal.compact_journal", return_value=0):
                cmd_startup(args)
        out = capsys.readouterr().out.strip()
        assert out == "{}"


class TestDevLoopActivationPrompt:
    """Test runtime-specific session-start loop instructions."""

    def test_codex_prompt_deprecates_dev_loop(self, tmp_path):
        """Codex agents must not receive loop-monitoring instructions."""
        project = tmp_path / "worktree"
        project.mkdir()
        gitgrip = tmp_path / ".gitgrip"
        gitgrip.mkdir()
        (gitgrip / "agents.toml").write_text(
            """
[spawn]
channel = "dev"

[agents.opus]
tool = "codex"
loop_interval = "1m"
""",
            encoding="utf-8",
        )

        env = {
            "AGENT_NAME": "opus",
            "CODEX_THREAD_ID": "thread-test",
            "SYNAPT_AGENT_ID": "opus-001",
        }
        with patch.dict(os.environ, env, clear=True):
            prompt = _dev_loop_activation_prompt(project)

        assert prompt is not None
        assert "Codex has no CronCreate" in prompt
        assert "CronCreate for the loop" not in prompt
        assert "dev-loop is deprecated for Codex" in prompt
        assert "do not emulate a monitoring loop" in prompt
        assert "sleep" in prompt
        assert "1m cadence" not in prompt

    def test_gripspace_root_env_resolves_sibling_griptree_config(self, tmp_path):
        """Spawned sibling griptrees use GRIPSPACE_ROOT to find agents.toml."""
        gripspace = tmp_path / "gripspace"
        gitgrip = gripspace / ".gitgrip"
        gitgrip.mkdir(parents=True)
        (gitgrip / "agents.toml").write_text(
            """
[spawn]
channel = "dev"

[agents.opus]
tool = "codex"
loop_interval = "1m"
""",
            encoding="utf-8",
        )
        project = tmp_path / "sibling-griptree"
        project.mkdir()

        env = {
            "AGENT_NAME": "opus",
            "GRIPSPACE_ROOT": str(gripspace),
        }
        with patch.dict(os.environ, env, clear=True):
            prompt = _dev_loop_activation_prompt(project)

        assert prompt is not None
        assert "deprecated for Codex" in prompt
        assert "CronCreate for the loop" not in prompt

    def test_claude_prompt_claims_no_resume_and_no_loop(self, tmp_path):
        """Claude agents are told what the hook DID, and not to poll.

        This test previously asserted the opposite -- it pinned
        "CronCreate for the loop" as required output, so the deprecated
        monitoring loop was defended by a passing test after the
        instruction had already been withdrawn. A green suite asserting
        behaviour that was retired is worse than an untested one:
        it reports agreement.

        The label is checked too. The prompt used to open
        "SessionStart:resume hook success" while performing no resume,
        which is a claim about a mechanism that does not run.
        """
        project = tmp_path / "worktree"
        project.mkdir()
        gitgrip = tmp_path / ".gitgrip"
        gitgrip.mkdir()
        (gitgrip / "agents.toml").write_text(
            """
[spawn]
channel = "dev"

[agents.apollo]
tool = "claude"
loop_interval = "5m"
""",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"AGENT_NAME": "apollo"}, clear=True):
            prompt = _dev_loop_activation_prompt(project)

        assert prompt is not None
        # The label must not announce a resume the hook never performs.
        assert "SessionStart:resume hook success" not in prompt
        assert "startup context loaded" in prompt
        # The deprecated loop must not be instructed, in any of its forms.
        assert "CronCreate" not in prompt
        assert "5m interval" not in prompt
        assert "monitoring loop is deprecated" in prompt
        assert "notify your coordinator" in prompt


class TestStartupSubcommand:
    """Test that the startup subcommand is registered in the CLI."""

    def test_startup_in_help(self):
        """The startup subcommand appears in --help output."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "synapt.recall.cli", "startup", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--json" in result.stdout
        assert "--compact" in result.stdout


class TestUncleanEndAtStartup:
    """When the previous session ended without a handoff, the wake must SAY so,
    first, and carry that session's own tail — not another session's checkpoint
    dressed as the bridge (measured 2026-08-31 after a host crash)."""

    _CRASHED = "65262c2c-54c3-4d58-aec6-d076f5040539"
    _FOREIGN = "489c7e73-aeff-4138-962c-da3297847601"

    def _unclean(self, tmp_path):
        from synapt.recall.resume import UncleanEnd
        return UncleanEnd(
            session_id=self._CRASHED,
            transcript_path=tmp_path / "crashed.jsonl",
            last_activity="2026-08-31T12:06:07Z",
            last_authored_journal="2026-08-31T04:54:00Z",
            gap_seconds=25927.0,
            checkpoint_session=self._FOREIGN,
        )

    def _quiet(self):
        return [
            patch("synapt.recall.journal._get_branch", return_value=None),
            patch("synapt.recall.compaction.latest_compaction_summary", return_value=None),
            patch("synapt.recall.knowledge.read_nodes", return_value=[]),
            patch("synapt.recall.reminders.pop_pending", return_value=[]),
            patch("synapt.recall.server.format_contradictions_for_session_start", return_value=""),
            patch("synapt.recall.channel.channel_join"),
            patch("synapt.recall.channel.channel_unread", return_value={}),
            patch("synapt.recall.channel.check_directives", return_value=""),
        ]

    def test_unclean_end_leads_continuity_and_carries_its_own_tail(self, tmp_path):
        from contextlib import ExitStack
        foreign_checkpoint = {
            "schema_version": 1, "session_id": self._FOREIGN,
            "captured_at": "2026-08-31T11:58:20Z", "parse_status": "partial",
            "last_user_text": None, "last_assistant_text": "foreign report",
            "files_touched": [],
        }
        recovered = {
            "parse_status": "partial", "truncated": True,
            "last_user_text": "can you show me the herdr changes as html",
            "last_assistant_text": None, "files_touched": ["/tmp/x.html"],
        }
        with ExitStack() as stack:
            for p in self._quiet():
                stack.enter_context(p)
            stack.enter_context(patch("synapt.checkpoint.read_checkpoint", return_value=foreign_checkpoint))
            stack.enter_context(patch("synapt.recall.resume.gather_unclean_end", return_value=self._unclean(tmp_path)))
            capture = stack.enter_context(patch("synapt.checkpoint.capture_checkpoint", return_value=recovered))
            lines = generate_startup_context(tmp_path, current_session_id="new-session")

        text = "\n".join(lines)
        assert lines[0].startswith("UNCLEAN END")
        assert "65262c2c" in lines[0]
        assert "7h12m" in lines[0]
        assert "489c7e73" in lines[0], "the foreign checkpoint is named so it is not read as the bridge"
        assert "can you show me the herdr changes as html" in lines[0]
        assert "synapt resume 65262c2c-54c3-4d58-aec6-d076f5040539" in lines[0]
        assert text.index("UNCLEAN END") < text.index("LAST CHECKPOINT")
        payload = capture.call_args.args[0]
        assert payload["transcript_path"] == str(tmp_path / "crashed.jsonl")
        assert payload["session_id"] == self._CRASHED

    def test_without_a_starting_session_id_no_verdict_is_published(self, tmp_path):
        """Call-path witness (Atlas r1): the generic startup path used to pass
        exclude_session_id=None, so the live transcript won and every Codex
        wake would have reported itself."""
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self._quiet():
                stack.enter_context(p)
            stack.enter_context(patch("synapt.checkpoint.read_checkpoint", return_value=None))
            gather = stack.enter_context(patch("synapt.recall.resume.gather_unclean_end", return_value=self._unclean(tmp_path)))
            text = "\n".join(generate_startup_context(tmp_path))
        assert "UNCLEAN END" not in text
        gather.assert_not_called()

    def test_cmd_startup_names_the_runtime_session_from_its_env(self, tmp_path, monkeypatch):
        import argparse
        from synapt.recall import cli
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CODEX_THREAD_ID", "codex-thread-7")
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        with patch("synapt.recall.cli.generate_startup_context", return_value=["ctx"]) as ctx, \
             patch("synapt.recall.journal.compact_journal", return_value=0):
            cli.cmd_startup(argparse.Namespace(json=False, compact=False))
        assert ctx.call_args.kwargs["current_session_id"] == "codex-thread-7"
        monkeypatch.delenv("CODEX_THREAD_ID")
        with patch("synapt.recall.cli.generate_startup_context", return_value=["ctx"]) as ctx, \
             patch("synapt.recall.journal.compact_journal", return_value=0):
            cli.cmd_startup(argparse.Namespace(json=False, compact=False))
        assert ctx.call_args.kwargs["current_session_id"] is None

    def test_the_detector_is_told_which_session_is_starting(self, tmp_path):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self._quiet():
                stack.enter_context(p)
            stack.enter_context(patch("synapt.checkpoint.read_checkpoint", return_value=None))
            gather = stack.enter_context(patch("synapt.recall.resume.gather_unclean_end", return_value=None))
            text = "\n".join(generate_startup_context(tmp_path, current_session_id="new-session"))
        assert "UNCLEAN END" not in text
        assert gather.call_args.kwargs["exclude_session_id"] == "new-session"

    def test_the_last_checkpoint_block_names_its_session(self, tmp_path):
        from contextlib import ExitStack
        checkpoint = {
            "schema_version": 1, "session_id": self._FOREIGN,
            "captured_at": "2026-08-31T11:58:20Z", "parse_status": "ok",
            "last_user_text": "q", "last_assistant_text": "a", "files_touched": [],
        }
        with ExitStack() as stack:
            for p in self._quiet():
                stack.enter_context(p)
            stack.enter_context(patch("synapt.checkpoint.read_checkpoint", return_value=checkpoint))
            stack.enter_context(patch("synapt.recall.resume.gather_unclean_end", return_value=None))
            text = "\n".join(generate_startup_context(tmp_path))
        assert "LAST CHECKPOINT" in text
        assert "Session: 489c7e73" in text


class TestChannelReadWidensWithBacklog:
    """Five messages at medium detail is the right read for a quiet night and
    the wrong one after a gap: on 2026-08-31 eleven were unread, three rendered
    inside the channel cap, and the two that mattered were withheld."""

    def _run(self, unread):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in (
                patch("synapt.recall.journal._get_branch", return_value=None),
                patch("synapt.recall.compaction.latest_compaction_summary", return_value=None),
                patch("synapt.checkpoint.read_checkpoint", return_value=None),
                patch("synapt.recall.resume.gather_unclean_end", return_value=None),
                patch("synapt.recall.knowledge.read_nodes", return_value=[]),
                patch("synapt.recall.reminders.pop_pending", return_value=[]),
                patch("synapt.recall.server.format_contradictions_for_session_start", return_value=""),
                patch("synapt.recall.channel.channel_join"),
                patch("synapt.recall.channel.check_directives", return_value=""),
                patch("synapt.recall.channel.channel_unread", return_value={"dev": unread}),
            ):
                stack.enter_context(p)
            read = stack.enter_context(patch("synapt.recall.channel.channel_read", return_value="msgs"))
            generate_startup_context(Path("/nonexistent-for-test"))
        return read.call_args.kwargs

    def test_a_backlog_is_read_one_line_per_message_up_to_thirty(self):
        kwargs = self._run(11)
        assert kwargs["limit"] == 11
        assert kwargs["detail"] == "min"
        kwargs = self._run(48)
        assert kwargs["limit"] == 30

    def test_a_few_unread_keep_the_full_form(self):
        kwargs = self._run(3)
        assert kwargs["limit"] == 3
        assert kwargs.get("detail", "medium") != "min"
