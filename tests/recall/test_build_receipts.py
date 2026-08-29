"""Durable MCP build receipts: #1018's demonstrated timeout path."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _release_server_markers():
    yield
    from synapt.recall import server

    server._release_build_server_markers()


class _FakeIndex:
    chunks = [object()]

    @staticmethod
    def stats() -> dict:
        return {"chunk_count": 7, "session_count": 3}


def _build_id(text: str) -> str:
    match = re.search(r"build_[0-9a-f]{12}", text)
    assert match, text
    return match.group(0)


def _wait_status(server, build_id: str, state: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        receipt = json.loads(server.recall_build_status(build_id))
        if receipt["state"] == state:
            return receipt
        time.sleep(0.01)
    raise AssertionError(server.recall_build_status(build_id))


def _running_receipt(build_id: str, pid: int) -> dict:
    timestamp = "2026-08-29T00:00:00+00:00"
    instance = "f" * 32
    return {
        "build_id": build_id,
        "state": "running",
        "phase": "indexing",
        "pid": pid,
        "server_instance": instance,
        "server_marker": f"build-server-{instance}.lock",
        "incremental": True,
        "created_at": timestamp,
        "started_at": timestamp,
        "updated_at": timestamp,
    }


def test_returns_receipt_before_completion_and_reuses_active_job(monkeypatch, tmp_path):
    from synapt.recall import cli, server

    entered = threading.Event()
    release = threading.Event()

    def blocked_build(project, *, use_embeddings, incremental, progress):
        progress("parsing")
        entered.set()
        assert release.wait(2)
        index_dir = server.project_index_dir(project)
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "data_002.db").write_bytes(b"new shard")
        return _FakeIndex()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_archive_and_build", blocked_build)
    monkeypatch.setattr(server, "_invalidate_cache", lambda: None)

    response = server.recall_build(incremental=False)
    build_id = _build_id(response)
    assert entered.wait(1), "worker did not start"
    running = json.loads(server.recall_build_status(build_id))
    assert running["state"] == "running"
    assert running["phase"] == "parsing"

    duplicate = server.recall_build(incremental=False)
    assert "already running" in duplicate.lower()
    assert _build_id(duplicate) == build_id

    release.set()
    completed = _wait_status(server, build_id, "completed")
    assert completed["stats"] == {"chunk_count": 7, "session_count": 3}
    assert completed["updated_shards"] == ["data_002.db"]


def test_same_process_status_does_not_reprobe_its_own_marker(monkeypatch, tmp_path):
    from synapt.recall import cli, server

    entered = threading.Event()
    release = threading.Event()

    def blocked_build(project, *, use_embeddings, incremental, progress):
        progress("parsing")
        entered.set()
        assert release.wait(2)
        return _FakeIndex()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_archive_and_build", blocked_build)
    monkeypatch.setattr(server, "_invalidate_cache", lambda: None)

    build_id = _build_id(server.recall_build())
    assert entered.wait(1), "worker did not start"
    monkeypatch.setattr(
        server,
        "_build_server_marker_alive",
        lambda *_args: pytest.fail("same-process marker was re-probed"),
    )

    running = json.loads(server.recall_build_status(build_id))
    assert running["state"] == "running"

    release.set()
    assert _wait_status(server, build_id, "completed")["result"] == "index built"


def test_receipt_pid_probe_reuses_query_only_windows_primitive(monkeypatch):
    from synapt.recall import server, session_start

    calls = []

    def forbidden_kill(*args):
        pytest.fail(f"os.kill{args} would terminate the process on Windows")

    monkeypatch.setattr(session_start.sys, "platform", "win32")
    monkeypatch.setattr(session_start.os, "kill", forbidden_kill)
    monkeypatch.setattr(
        session_start,
        "_pid_alive_win32",
        lambda pid: calls.append(pid) or True,
    )

    assert server._pid_alive is session_start._pid_alive
    assert server._pid_alive(424242) is True
    assert calls == [424242]


def test_status_cannot_overwrite_a_terminal_receipt_with_interrupted(monkeypatch, tmp_path):
    from synapt.recall import cli, server

    entered = threading.Event()
    release = threading.Event()
    original_owner_alive = server._receipt_owner_alive

    def blocked_build(project, *, use_embeddings, incremental, progress):
        progress("parsing")
        entered.set()
        assert release.wait(2)
        return None

    def delayed_owner_check(project, receipt):
        release.set()
        time.sleep(0.1)
        return original_owner_alive(project, receipt)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_archive_and_build", blocked_build)
    monkeypatch.setattr(server, "_invalidate_cache", lambda: None)

    build_id = _build_id(server.recall_build())
    assert entered.wait(1), "worker did not start"
    monkeypatch.setattr(server, "_receipt_owner_alive", delayed_owner_check)

    observed = json.loads(server.recall_build_status(build_id))
    assert observed["state"] == "running"
    completed = _wait_status(server, build_id, "completed")
    assert completed["result"] == "no transcripts found"


def test_failure_is_a_terminal_durable_receipt(monkeypatch, tmp_path):
    from synapt.recall import cli, server

    def failed_build(project, *, use_embeddings, incremental, progress):
        progress("indexing")
        raise RuntimeError("fixture boom")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_archive_and_build", failed_build)
    monkeypatch.setattr(server, "_invalidate_cache", lambda: None)

    build_id = _build_id(server.recall_build())
    failed = _wait_status(server, build_id, "failed")
    assert failed["phase"] == "failed"
    assert failed["error"] == "RuntimeError: fixture boom"


def test_worker_start_failure_rewrites_prewritten_receipt_as_failed(monkeypatch, tmp_path):
    from synapt.recall import server

    def fail_to_start(self):
        raise RuntimeError("thread fixture")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(threading.Thread, "start", fail_to_start)

    response = server.recall_build()
    build_id = _build_id(response)
    receipt = json.loads(server.recall_build_status(build_id))
    assert response.startswith("Build not started:")
    assert receipt["state"] == "failed"
    assert receipt["error"] == "worker did not start: RuntimeError: thread fixture"


def test_lock_timeout_is_failed_not_reported_as_empty(monkeypatch, tmp_path):
    from synapt.recall import cli, server

    def lock_timeout(project, *, use_embeddings, incremental, progress):
        progress("lock_timeout")
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_archive_and_build", lock_timeout)
    monkeypatch.setattr(server, "_invalidate_cache", lambda: None)

    build_id = _build_id(server.recall_build())
    result = _wait_status(server, build_id, "failed")
    assert "timed out waiting for the build lock" in result["error"]


def test_no_transcripts_is_a_completed_empty_build(monkeypatch, tmp_path):
    from synapt.recall import cli, server

    def empty_build(project, *, use_embeddings, incremental, progress):
        progress("parsing")
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_archive_and_build", empty_build)
    monkeypatch.setattr(server, "_invalidate_cache", lambda: None)

    build_id = _build_id(server.recall_build())
    result = _wait_status(server, build_id, "completed")
    assert result["result"] == "no transcripts found"
    assert result["stats"] == {"chunk_count": 0, "session_count": 0}


def test_cache_invalidation_warning_preserves_terminal_receipt(monkeypatch, tmp_path):
    from synapt.recall import cli, server

    def build(project, *, use_embeddings, incremental, progress):
        progress("finalizing")
        return _FakeIndex()

    def invalidation_failure():
        raise RuntimeError("cache fixture")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_archive_and_build", build)
    monkeypatch.setattr(server, "_invalidate_cache", invalidation_failure)

    build_id = _build_id(server.recall_build())
    result = _wait_status(server, build_id, "completed")
    assert result["cache_warning"] == "RuntimeError: cache fixture"


def test_dead_process_receipt_becomes_interrupted(monkeypatch, tmp_path):
    from synapt.recall import server

    monkeypatch.chdir(tmp_path)
    receipt = _running_receipt("build_0123456789ab", 999_999_999)
    receipt["phase"] = "clustering"
    server._write_build_receipt(tmp_path.resolve(), receipt)

    result = json.loads(server.recall_build_status(receipt["build_id"]))
    assert result["state"] == "interrupted"
    assert result["phase"] == "interrupted"
    assert "exited before writing a terminal receipt" in result["error"]


def test_same_pid_from_previous_server_instance_is_interrupted(monkeypatch, tmp_path):
    from synapt.recall import server

    monkeypatch.chdir(tmp_path)
    receipt = _running_receipt("build_123456789abc", os.getpid())
    receipt["phase"] = "clustering"
    receipt["server_instance"] = "e" * 32
    receipt["server_marker"] = f"build-server-{'e' * 32}.lock"
    server._write_build_receipt(tmp_path.resolve(), receipt)

    result = json.loads(server.recall_build_status(receipt["build_id"]))
    assert result["state"] == "interrupted"


def test_corrupt_existing_receipt_refuses_a_new_build(monkeypatch, tmp_path):
    from synapt.recall import server

    monkeypatch.chdir(tmp_path)
    directory = server._build_receipts_dir(tmp_path.resolve())
    directory.mkdir(parents=True)
    (directory / "build_deadbeefcafe.json").write_text("{not json", encoding="utf-8")

    response = server.recall_build()
    assert response.startswith("Build not started:")
    assert "receipt is unreadable" in response
    assert "Remove that file to allow new builds" in response


@pytest.mark.parametrize(
    ("filename", "build_id", "server_marker"),
    [
        (
            "build_deadbeefcafe.json",
            "../../../escaped",
            "build-server-0123456789abcdef0123456789abcdef.lock",
        ),
        (
            "build_deadbeefcafe.json",
            "build_0123456789ab",
            "build-server-0123456789abcdef0123456789abcdef.lock",
        ),
        (
            "build_deadbeefcafe.json",
            "build_deadbeefcafe",
            "../../marker-target.lock",
        ),
        ("build_deadbeefcafe.json", "build_deadbeefcafe", None),
    ],
)
def test_invalid_active_receipt_envelope_refuses_without_side_effects(
    monkeypatch, tmp_path, filename, build_id, server_marker,
):
    from synapt.recall import server

    monkeypatch.chdir(tmp_path)
    directory = server._build_receipts_dir(tmp_path.resolve())
    directory.mkdir(parents=True)
    receipt = _running_receipt(build_id, os.getpid())
    if server_marker is None:
        receipt.pop("server_marker")
    else:
        receipt["server_marker"] = server_marker
    original = json.dumps(receipt)
    (directory / filename).write_text(original, encoding="utf-8")

    response = server.recall_build()

    assert response.startswith("Build not started:")
    assert "receipt is unreadable" in response
    assert (directory / filename).read_text(encoding="utf-8") == original
    assert not (tmp_path / "escaped.json").exists()
    assert not (tmp_path / "marker-target.lock").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", "runnning"),
        ("state", ["running"]),
        ("phase", "completed"),
        ("phase", "banana"),
        ("pid", "12345"),
        ("server_instance", None),
        ("server_instance", "not-a-server-instance"),
        ("created_at", "not-a-time"),
        ("updated_at", "2026-08-29T00:00:00"),
        ("started_at", "not-a-time"),
    ],
)
def test_invalid_receipt_core_refuses_without_rewrite_or_duplicate(
    monkeypatch, tmp_path, field, value,
):
    from synapt.recall import server

    monkeypatch.chdir(tmp_path)
    directory = server._build_receipts_dir(tmp_path.resolve())
    directory.mkdir(parents=True)
    path = directory / "build_deadbeefcafe.json"
    receipt = _running_receipt("build_deadbeefcafe", os.getpid())
    if value is None:
        receipt.pop(field)
    else:
        receipt[field] = value
    original = json.dumps(receipt)
    path.write_text(original, encoding="utf-8")

    response = server.recall_build()

    assert response.startswith("Build not started:")
    assert "receipt is unreadable" in response
    assert path.read_text(encoding="utf-8") == original
    assert list(directory.glob("build_*.json")) == [path]


@pytest.mark.parametrize(
    ("field", "value", "error_field"),
    [
        ("updated_shards", None, "updated_shards"),
        ("finished_at", "not-a-time", "finished_at"),
    ],
)
def test_status_refuses_incomplete_terminal_receipt(
    monkeypatch, tmp_path, field, value, error_field,
):
    from synapt.recall import server

    monkeypatch.chdir(tmp_path)
    directory = server._build_receipts_dir(tmp_path.resolve())
    directory.mkdir(parents=True)
    path = directory / "build_deadbeefcafe.json"
    receipt = _running_receipt("build_deadbeefcafe", os.getpid())
    receipt.update({
        "state": "completed",
        "phase": "completed",
        "finished_at": "2026-08-29T00:01:00+00:00",
        "updated_shards": [],
        "stats": {"chunk_count": 0, "session_count": 0},
        "result": "no transcripts found",
    })
    receipt.pop("started_at")
    if value is None:
        receipt.pop(field)
    else:
        receipt[field] = value
    original = json.dumps(receipt)
    path.write_text(original, encoding="utf-8")

    response = server.recall_build_status("build_deadbeefcafe")

    assert response.startswith("Build receipt unreadable:")
    assert error_field in response
    assert path.read_text(encoding="utf-8") == original


def test_live_sibling_process_receipt_is_reused(monkeypatch, tmp_path):
    from synapt.recall import server

    monkeypatch.chdir(tmp_path)
    marker = "build-server-abcdef0123456789abcdef0123456789.lock"
    source_root = str(Path(server.__file__).resolve().parents[2])
    script = (
        "import time; from pathlib import Path; "
        "from synapt.recall.cli import _acquire_build_lock; "
        "fd = _acquire_build_lock(Path.cwd() / '.synapt' / 'recall', timeout=0, "
        f"name={marker!r}); assert fd is not None; print('ready', flush=True); time.sleep(10)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = source_root
    child = subprocess.Popen(
        [sys.executable, "-c", script], cwd=tmp_path, env=environment,
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        receipt = _running_receipt("build_abcdef012345", child.pid)
        receipt["server_instance"] = marker.removeprefix("build-server-").removesuffix(".lock")
        receipt["server_marker"] = marker
        server._write_build_receipt(tmp_path.resolve(), receipt)
        response = server.recall_build()
        assert "already running" in response.lower()
        assert _build_id(response) == receipt["build_id"]
    finally:
        child.terminate()
        child.wait(timeout=3)

    # Simulate PID reuse by an unrelated live process. Its numeric liveness
    # cannot replace the original server-marker lock as provenance.
    replacement = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        receipt["pid"] = replacement.pid
        server._write_build_receipt(tmp_path.resolve(), receipt)
        result = json.loads(server.recall_build_status(receipt["build_id"]))
        assert result["state"] == "interrupted"
    finally:
        replacement.terminate()
        replacement.wait(timeout=3)


def test_live_mismatched_marker_pair_refuses_without_duplicate(monkeypatch, tmp_path):
    from synapt.recall import server

    monkeypatch.chdir(tmp_path)
    held_instance = "a" * 32
    marker = f"build-server-{held_instance}.lock"
    source_root = str(Path(server.__file__).resolve().parents[2])
    script = (
        "import time; from pathlib import Path; "
        "from synapt.recall.cli import _acquire_build_lock; "
        "fd = _acquire_build_lock(Path.cwd() / '.synapt' / 'recall', timeout=0, "
        f"name={marker!r}); assert fd is not None; print('ready', flush=True); time.sleep(10)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = source_root
    child = subprocess.Popen(
        [sys.executable, "-c", script], cwd=tmp_path, env=environment,
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        directory = server._build_receipts_dir(tmp_path.resolve())
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "build_deadbeefcafe.json"
        receipt = _running_receipt("build_deadbeefcafe", child.pid)
        receipt["server_instance"] = "b" * 32
        receipt["server_marker"] = marker
        original = json.dumps(receipt)
        path.write_text(original, encoding="utf-8")

        response = server.recall_build()

        assert response.startswith("Build not started:")
        assert "server_marker does not match server_instance" in response
        assert path.read_text(encoding="utf-8") == original
        assert list(directory.glob("build_*.json")) == [path]
    finally:
        child.terminate()
        child.wait(timeout=3)


def test_changed_shards_excludes_unchanged_existing_store(monkeypatch, tmp_path):
    from synapt.recall import cli, server

    index_dir = server.project_index_dir(tmp_path.resolve())
    index_dir.mkdir(parents=True)
    (index_dir / "data_001.db").write_bytes(b"unchanged")

    def build(project, *, use_embeddings, incremental, progress):
        progress("indexing")
        (index_dir / "data_002.db").write_bytes(b"changed")
        return _FakeIndex()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_archive_and_build", build)
    monkeypatch.setattr(server, "_invalidate_cache", lambda: None)

    build_id = _build_id(server.recall_build())
    result = _wait_status(server, build_id, "completed")
    assert result["updated_shards"] == ["data_002.db"]


def test_changed_shards_detects_wal_only_change(monkeypatch, tmp_path):
    from synapt.recall import cli, server

    index_dir = server.project_index_dir(tmp_path.resolve())
    index_dir.mkdir(parents=True)
    base = index_dir / "data_001.db"
    changed_wal = index_dir / "data_001.db-wal"
    unchanged = index_dir / "data_002.db"
    unchanged_wal = index_dir / "data_002.db-wal"
    base.write_bytes(b"unchanged base")
    changed_wal.write_bytes(b"before")
    unchanged.write_bytes(b"unchanged base")
    unchanged_wal.write_bytes(b"unchanged wal")

    def build(project, *, use_embeddings, incremental, progress):
        progress("indexing")
        changed_wal.write_bytes(b"after with a different size")
        return _FakeIndex()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_archive_and_build", build)
    monkeypatch.setattr(server, "_invalidate_cache", lambda: None)

    build_id = _build_id(server.recall_build())
    result = _wait_status(server, build_id, "completed")
    assert result["updated_shards"] == ["data_001.db"]


def test_cli_build_forwards_phase_callback_without_changing_sync_result(monkeypatch, tmp_path):
    from synapt.recall import cli

    phases = []
    expected = object()
    monkeypatch.setattr(cli, "_acquire_build_lock", lambda data_dir: 42)
    monkeypatch.setattr(cli, "_release_build_lock", lambda fd: None)

    def inner(*args):
        args[-1]("parsing")
        return expected

    monkeypatch.setattr(cli, "_archive_and_build_locked", inner)
    result = cli._archive_and_build(tmp_path, progress=phases.append)
    assert result is expected
    assert phases == ["waiting_for_lock", "parsing"]


def test_status_tool_is_registered():
    from synapt.recall import server

    registered = []

    class FakeMCP:
        def tool(self):
            def decorate(fn):
                registered.append(getattr(fn, "__name__", repr(fn)))
                return fn
            return decorate

    server.register_tools(FakeMCP())
    assert "recall_build" in registered
    assert "recall_build_status" in registered
