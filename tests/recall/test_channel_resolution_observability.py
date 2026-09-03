"""Every channel operation must name the state store it resolved.

The operational failure behind recall#996 returned a confident membership
error from one ``channels.db`` while the intended shared store already held
the membership. Store provenance belongs at the common MCP boundary so new
actions inherit it automatically. Caller identity is deliberately absent:
answering who an agent is belongs to the premium identity layer.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _isolation_helpers import owned_store


class TestChannelResolutionObservability(unittest.TestCase):
    def setUp(self) -> None:
        self._store = owned_store()
        self._previous_agent_id = os.environ.get("SYNAPT_AGENT_ID")
        os.environ["SYNAPT_AGENT_ID"] = "sentinel-witness"

        from synapt.recall.actions import reset_action_registry

        reset_action_registry()

    def tearDown(self) -> None:
        from synapt.recall.actions import reset_action_registry

        reset_action_registry()
        if self._previous_agent_id is None:
            os.environ.pop("SYNAPT_AGENT_ID", None)
        else:
            os.environ["SYNAPT_AGENT_ID"] = self._previous_agent_id
        self._store.restore()

    @property
    def expected_state_db(self) -> Path:
        return (
            self._store.root
            / "recall-root"
            / ".synapt"
            / "recall"
            / "channels"
            / "channels.db"
        ).resolve()

    def test_who_names_store_without_inventing_caller_identity(self) -> None:
        from synapt.recall.server import recall_channel

        result = recall_channel(action="who")

        self.assertIn(f"Channel state store: {self.expected_state_db}", result)
        self.assertIn("No agents online.", result)
        self.assertNotIn("Caller identity:", result)

    def test_unread_names_store_on_membership_error_without_identity(self) -> None:
        from synapt.recall.server import recall_channel

        result = recall_channel(action="unread", detail="low")

        self.assertIn(f"Channel state store: {self.expected_state_db}", result)
        self.assertIn("No channel memberships", result)
        self.assertNotIn("Caller identity:", result)

    def test_common_boundary_covers_every_registered_action(self) -> None:
        from synapt.recall.actions import ActionRegistry, get_action_registry
        from synapt.recall.server import recall_channel

        state_db = Path("/distinct/witness/channels.db")
        resolved_state_db = state_db.resolve()
        log_dir = Path("/distinct/witness/log-store")
        resolved_log_dir = log_dir.resolve()
        actions = sorted(get_action_registry().known_actions)
        with (
            patch("synapt.recall.channel._db_path", return_value=state_db),
            patch("synapt.recall.channel._channels_dir", return_value=log_dir),
            patch.object(ActionRegistry, "dispatch", return_value="payload") as dispatch,
        ):
            for action in actions:
                with self.subTest(action=action):
                    result = recall_channel(action=action)
                    self.assertEqual(
                        result,
                        f"Channel state store: {resolved_state_db} "
                        f"(source: env:SYNAPT_RECALL_ROOT)\n"
                        f"Channel log store: {resolved_log_dir} "
                        f"(source: env:SYNAPT_RECALL_ROOT)\npayload",
                    )

        self.assertEqual(dispatch.call_count, len(actions))

    def test_dispatch_failure_still_names_resolved_store(self) -> None:
        from synapt.recall.actions import ActionRegistry
        from synapt.recall.server import recall_channel

        state_db = Path("/distinct/failure/channels.db")
        resolved_state_db = state_db.resolve()
        log_dir = Path("/distinct/failure/log-store")
        resolved_log_dir = log_dir.resolve()
        with (
            patch("synapt.recall.channel._db_path", return_value=state_db),
            patch("synapt.recall.channel._channels_dir", return_value=log_dir),
            patch.object(ActionRegistry, "dispatch", side_effect=RuntimeError("boom")),
        ):
            result = recall_channel(action="read")

        self.assertEqual(
            result,
            f"Channel state store: {resolved_state_db} "
            f"(source: env:SYNAPT_RECALL_ROOT)\n"
            f"Channel log store: {resolved_log_dir} "
            f"(source: env:SYNAPT_RECALL_ROOT)\nChannel failed: boom",
        )

    def test_store_resolution_failure_reports_the_original_error(self) -> None:
        from synapt.recall.server import recall_channel

        with patch(
            "synapt.recall.channel._db_path",
            side_effect=RuntimeError("root unavailable"),
        ):
            result = recall_channel(action="read")

        self.assertEqual(result, "Channel failed: root unavailable")


class TestLogStoreDivergesFromStateStore(unittest.TestCase):
    """The log directory (Tier-2 global, when a gripspace's manifest
    resolves) and the state db directory (ALWAYS Tier-3 local, by
    channel._db_path's own documented design -- "presence, cursors, pins,
    and mutes are per-gripspace even when channels are shared") can be two
    different directories for the very same channel, from the very same cwd.

    This is the mechanism behind a reported production symptom: a real
    gripspace-local dev.jsonl "stayed behind looking live" after the actual
    live log moved to the org/project Tier-2 path -- recall_channel kept
    printing only the (correctly local) state store, so nothing in its
    output ever told a reader the log itself lives somewhere else. The
    resolver divergence recall#916/#1071 fixed was cwd-vs-GRIPSPACE_ROOT
    disagreement; this is a structural divergence between two DIFFERENT
    resolvers (_channels_dir vs _db_path) that GRIPSPACE_ROOT precedence
    does not touch, because it was never a bug in either resolver
    individually -- _db_path's local-only behavior is deliberate.
    """

    def setUp(self) -> None:
        from synapt.recall.actions import reset_action_registry
        from synapt.recall.core import _gripspace_cache

        self._tmp = tempfile.mkdtemp()
        tmp_path = Path(self._tmp)
        self._fake_home = tmp_path / "home"
        self._fake_home.mkdir()

        self._grip_root = tmp_path / "gripspace"
        (self._grip_root / ".gitgrip" / "spaces" / "main").mkdir(parents=True)
        (self._grip_root / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
        (self._grip_root / ".gitgrip" / "spaces" / "main" / "gripspace.yml").write_text(
            "manifest:\n  url: git@github.com:synapt-dev/synapt-gripspace.git\n"
        )

        self._previous_cwd = os.getcwd()
        os.chdir(self._grip_root)

        self._env_vars = (
            "SYNAPT_RECALL_ROOT",
            "SYNAPT_SHARED_CHANNELS_DIR",
            "GRIPSPACE_ROOT",
            "SYNAPT_AGENT_ID",
        )
        self._previous_env = {name: os.environ.get(name) for name in self._env_vars}
        for name in self._env_vars:
            os.environ.pop(name, None)
        os.environ["SYNAPT_AGENT_ID"] = "sentinel-witness"

        self._home_patch = patch(
            "synapt.recall.core.Path.home", return_value=self._fake_home
        )
        self._home_patch.start()
        _gripspace_cache.clear()

        reset_action_registry()

    def tearDown(self) -> None:
        from synapt.recall.actions import reset_action_registry
        from synapt.recall.core import _gripspace_cache

        reset_action_registry()
        self._home_patch.stop()
        _gripspace_cache.clear()
        os.chdir(self._previous_cwd)
        for name, prev in self._previous_env.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_state_store_and_log_store_can_diverge_and_both_must_be_named(self) -> None:
        from synapt.recall.channel import _channels_dir, _db_path
        from synapt.recall.server import recall_channel

        log_store = _channels_dir().resolve()
        state_store = _db_path().resolve()

        # Sanity: this scenario only demonstrates the defect if the two
        # resolvers actually land in different directories -- Tier-2 global
        # for the log, Tier-3 local (by design) for the state db.
        self.assertNotEqual(
            log_store,
            state_store.parent,
            "fixture did not reproduce the divergence -- log store and "
            "state store already agree, so this test cannot demonstrate "
            "the reported failure mode",
        )

        result = recall_channel(action="who")

        self.assertIn(f"Channel state store: {state_store}", result)
        self.assertIn(
            f"Channel log store: {log_store}",
            result,
            "the live JSONL log's directory diverged from the printed "
            "state store and recall_channel's output never said so -- "
            "exactly the reported failure mode: a reader trusts the "
            "printed path as the whole picture and never learns the log "
            "lives elsewhere",
        )


class TestCoordinateResolutionHonorsSameOverride(unittest.TestCase):
    """SYNAPT_RECALL_ROOT is an explicit, must-exist override that
    ``project_data_dir()`` (backing ``_db_path`` / the Tier-3 local
    fallback) honors outright -- but ``_read_manifest_url()`` (backing
    ``_channels_dir``'s Tier-2 global attempt) never consulted it at all,
    even though both resolvers already share ``_find_gripspace_root`` for
    the plain cwd/GRIPSPACE_ROOT case (recall#1071).

    So a caller who sets SYNAPT_RECALL_ROOT to redirect the local store
    (a reconstruction scratch dir, an export --path, a test isolation
    helper) got exactly the divergence a reported production symptom
    made visible: the log jumps to the real gripspace's Tier-2 global store
    while the state db honors the override and lands somewhere else
    entirely -- not merely "two deliberately different stores for one
    gripspace" (that part is by design) but two DIFFERENT gripspaces.

    This test sits inside a real, resolvable gripspace (manifest present)
    and sets SYNAPT_RECALL_ROOT to a sibling directory with no manifest of
    its own. Both resolvers must agree that this override selects a
    coordinate with no Tier-2 store available, landing both stores under
    the SAME override directory -- one resolver, one coordinate, for both.
    """

    def setUp(self) -> None:
        from synapt.recall.actions import reset_action_registry
        from synapt.recall.core import _gripspace_cache

        self._tmp = tempfile.mkdtemp()
        tmp_path = Path(self._tmp)
        self._fake_home = tmp_path / "home"
        self._fake_home.mkdir()

        self._grip_root = tmp_path / "gripspace"
        (self._grip_root / ".gitgrip" / "spaces" / "main").mkdir(parents=True)
        (self._grip_root / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
        (self._grip_root / ".gitgrip" / "spaces" / "main" / "gripspace.yml").write_text(
            "manifest:\n  url: git@github.com:synapt-dev/synapt-gripspace.git\n"
        )

        self._override_root = tmp_path / "override-scratch"
        self._override_root.mkdir()

        self._previous_cwd = os.getcwd()
        os.chdir(self._grip_root)

        self._env_vars = (
            "SYNAPT_RECALL_ROOT",
            "SYNAPT_SHARED_CHANNELS_DIR",
            "GRIPSPACE_ROOT",
            "SYNAPT_AGENT_ID",
        )
        self._previous_env = {name: os.environ.get(name) for name in self._env_vars}
        for name in self._env_vars:
            os.environ.pop(name, None)
        os.environ["SYNAPT_AGENT_ID"] = "sentinel-witness"
        os.environ["SYNAPT_RECALL_ROOT"] = str(self._override_root)

        self._home_patch = patch(
            "synapt.recall.core.Path.home", return_value=self._fake_home
        )
        self._home_patch.start()
        _gripspace_cache.clear()

        reset_action_registry()

    def tearDown(self) -> None:
        from synapt.recall.actions import reset_action_registry
        from synapt.recall.core import _gripspace_cache

        reset_action_registry()
        self._home_patch.stop()
        _gripspace_cache.clear()
        os.chdir(self._previous_cwd)
        for name, prev in self._previous_env.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_synapt_recall_root_override_is_honored_by_both_stores(self) -> None:
        from synapt.recall.channel import _channels_dir, _db_path

        log_store = _channels_dir().resolve()
        state_store = _db_path().resolve()

        self.assertEqual(
            state_store.parent,
            self._override_root.resolve() / ".synapt" / "recall" / "channels",
            "sanity: the state store must land under the SYNAPT_RECALL_ROOT "
            "override -- if this fails, project_data_dir's own override "
            "handling changed and this fixture no longer isolates the "
            "coordinate divergence this test targets",
        )
        self.assertEqual(
            log_store,
            state_store.parent,
            "the log store did not honor SYNAPT_RECALL_ROOT: it resolved "
            "Tier-2 against the REAL gripspace on cwd/GRIPSPACE_ROOT while "
            "the state store correctly followed the override -- one "
            "resolver did not decide the coordinate for both",
        )


class TestOrphanedLocalChannelStoreIsReported(unittest.TestCase):
    """A reported production symptom: a gripspace-local dev.jsonl that
    stayed behind looking live after the real log moved to the org/project
    Tier-2 global store, with channels.db sitting right beside it (state
    db is Tier-3 local by design, so it is genuinely still being written)
    — nothing told a reader the JSONL log itself was stale and orphaned.

    The disclosure half (recall PR #1077) made the CURRENT resolution
    visible (both stores named on every call). This test targets the
    detect-and-report half: when a local channels dir already holds a
    channel JSONL file at the same time this call's log resolves Tier-2
    global, that is exactly the leftover shape — report it, never delete
    it.
    """

    def setUp(self) -> None:
        from synapt.recall.actions import reset_action_registry
        from synapt.recall.core import _gripspace_cache

        self._tmp = tempfile.mkdtemp()
        tmp_path = Path(self._tmp)
        self._fake_home = tmp_path / "home"
        self._fake_home.mkdir()

        self._grip_root = tmp_path / "gripspace"
        (self._grip_root / ".gitgrip" / "spaces" / "main").mkdir(parents=True)
        (self._grip_root / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
        (self._grip_root / ".gitgrip" / "spaces" / "main" / "gripspace.yml").write_text(
            "manifest:\n  url: git@github.com:synapt-dev/synapt-gripspace.git\n"
        )

        self._previous_cwd = os.getcwd()
        os.chdir(self._grip_root)

        self._env_vars = (
            "SYNAPT_RECALL_ROOT",
            "SYNAPT_SHARED_CHANNELS_DIR",
            "GRIPSPACE_ROOT",
            "SYNAPT_AGENT_ID",
        )
        self._previous_env = {name: os.environ.get(name) for name in self._env_vars}
        for name in self._env_vars:
            os.environ.pop(name, None)
        os.environ["SYNAPT_AGENT_ID"] = "sentinel-witness"

        self._home_patch = patch(
            "synapt.recall.core.Path.home", return_value=self._fake_home
        )
        self._home_patch.start()
        _gripspace_cache.clear()

        reset_action_registry()

    def tearDown(self) -> None:
        from synapt.recall.actions import reset_action_registry
        from synapt.recall.core import _gripspace_cache

        reset_action_registry()
        self._home_patch.stop()
        _gripspace_cache.clear()
        os.chdir(self._previous_cwd)
        for name, prev in self._previous_env.items():
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_stale_local_jsonl_is_reported_when_log_moved_global(self) -> None:
        from synapt.recall.channel import _channels_dir, _local_channels_dir
        from synapt.recall.server import recall_channel

        # Sanity: this call's log resolves Tier-2 global.
        log_store = _channels_dir().resolve()
        self.assertNotEqual(
            log_store,
            _local_channels_dir().resolve(),
            "fixture did not reproduce Tier-2 resolution -- cannot "
            "demonstrate the leftover-local-file failure mode",
        )

        # Seed the leftover: a channel JSONL sitting in the now-orphaned
        # local dir, with the state db already beside it (created above by
        # the first call to _db_path/_channels_dir's mkdir side effects on
        # the global tier, so the local dir does not exist yet -- create it
        # by hand, exactly as an old local-fallback write would have left
        # it before the manifest started resolving Tier-2).
        local_dir = _local_channels_dir()
        local_dir.mkdir(parents=True, exist_ok=True)
        stale_log = local_dir / "dev.jsonl"
        stale_log.write_text('{"channel": "dev", "text": "stale leftover"}\n')
        (local_dir / "channels.db").touch()

        result = recall_channel(action="who")

        # The disclosure fix (recall PR #1077) already names the STATE
        # STORE's directory unconditionally, and that directory happens to
        # equal the local dir this test seeded -- so asserting the bare
        # directory path appears would pass vacuously with no orphan
        # detection at all. The orphan signal has to be something that
        # line does NOT already carry: the stale JSONL FILE's own path,
        # or an explicit label distinguishing "leftover" from "the state
        # store I am using right now".
        self.assertIn(
            str(stale_log),
            result,
            "a leftover local channels dir with a real JSONL file existed "
            "at the same time this call's log resolved Tier-2 global, and "
            "nothing in recall_channel's output named the leftover FILE "
            "itself (only the directory, via the pre-existing state-store "
            "line) -- exactly the reported symptom: a stale file "
            "with a live-looking name and no signal that it is not the "
            "live log",
        )
        self.assertIn(
            "orphan",
            result.lower(),
            "the report must be distinguishable from the ordinary "
            "state-store disclosure line, or a reader cannot tell this "
            "path is a leftover rather than the store currently in use",
        )
        self.assertTrue(
            stale_log.exists(),
            "the leftover file must never be deleted by a detection pass",
        )

    def test_no_report_when_nothing_is_orphaned(self) -> None:
        from synapt.recall.server import recall_channel

        result = recall_channel(action="who")

        self.assertNotIn("orphan", result.lower())

    def test_local_dir_with_only_state_db_and_no_jsonl_is_not_reported(self) -> None:
        """A local dir that exists (holding only channels.db, as it does
        once anything opens the state db there) but has never held a
        channel JSONL is NOT a leftover -- there is no log to have gone
        stale, only the state db that is supposed to live there. The
        prior version of this suite never constructed this exact shape,
        so a mutant that reports an orphan
        purely because the DIRECTORY exists (dropping the "*.jsonl"
        condition down to an unconditional True) survived undetected.
        """
        from synapt.recall.channel import _channels_dir, _local_channels_dir
        from synapt.recall.server import recall_channel

        # The local dir exists (as it would once something opens the
        # state db there) but was never used to write a channel JSONL,
        # while Tier-2 still resolves for the log -- "a local dir exists,
        # and the log for this call comes from elsewhere".
        local_dir = _local_channels_dir().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "channels.db").touch()
        log_store = _channels_dir().resolve()
        self.assertNotEqual(
            log_store,
            local_dir,
            "fixture did not reproduce Tier-2 resolution -- cannot "
            "demonstrate the no-jsonl negative case",
        )
        self.assertTrue(local_dir.is_dir(), "state db mkdir side effect did not create the local dir")
        self.assertEqual(
            list(local_dir.glob("*.jsonl")),
            [],
            "fixture leaked a jsonl file into the local dir -- this test "
            "targets the case where NONE exists",
        )

        result = recall_channel(action="who")

        self.assertNotIn(
            "orphan",
            result.lower(),
            "a local dir with no channel JSONL was reported as an "
            "orphaned leftover -- there was never a log there to go "
            "stale, only the state db that belongs there by design",
        )


class TestDisclosureLinesNameTheRootSource(unittest.TestCase):
    """recall#936, item 2's marker-persistence follow-on: the existing
    store-disclosure lines name WHICH store resolved, but not WHAT chose
    the coordinate. A marker converging a bare CLI call onto an env-bound
    call's root is invisible unless the disclosure line also says so.
    """

    def setUp(self) -> None:
        self._store = owned_store()
        self._previous_agent_id = os.environ.get("SYNAPT_AGENT_ID")
        os.environ["SYNAPT_AGENT_ID"] = "sentinel-witness"

        from synapt.recall.actions import reset_action_registry

        reset_action_registry()

    def tearDown(self) -> None:
        from synapt.recall.actions import reset_action_registry

        reset_action_registry()
        if self._previous_agent_id is None:
            os.environ.pop("SYNAPT_AGENT_ID", None)
        else:
            os.environ["SYNAPT_AGENT_ID"] = self._previous_agent_id
        self._store.restore()

    def test_channel_state_store_line_names_the_env_source(self) -> None:
        from synapt.recall.server import recall_channel

        # owned_store() sets SYNAPT_RECALL_ROOT, so both stores resolve
        # via that override deterministically.
        result = recall_channel(action="who")
        lines = result.splitlines()
        state_line = next(l for l in lines if l.startswith("Channel state store:"))
        self.assertIn(
            "env:SYNAPT_RECALL_ROOT",
            state_line,
            "the state-store disclosure line must name the source that "
            "chose its root, not only the root itself",
        )


if __name__ == "__main__":
    unittest.main()
