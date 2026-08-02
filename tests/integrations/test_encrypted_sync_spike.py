"""Executable contract for the encrypted-sync Spike A demo.

The fast tests pin each property independently.  The Docker test composes
those properties and is opt-in because it builds images and starts containers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SPIKE_ROOT = REPO_ROOT / "demo" / "encrypted_sync"
COMPOSE_PATH = SPIKE_ROOT / "compose.yaml"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_docker_context_excludes_local_state_and_unrelated_demo_assets() -> None:
    rules = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert rules[0] == "**"
    assert "!src/**" in rules
    assert "!demo/encrypted_sync/**" in rules
    assert "!demo/**" not in rules


def test_compose_has_exactly_two_agents_and_one_blind_relay() -> None:
    services = _compose()["services"]

    assert set(services) == {"agent-a", "agent-b", "relay"}
    assert services["agent-a"]["networks"] == ["agent_a_relay"]
    assert services["agent-b"]["networks"] == ["agent_b_relay"]
    assert set(services["relay"]["networks"]) == {
        "agent_a_relay",
        "agent_b_relay",
    }


def test_compose_gives_agents_no_direct_network_or_shared_data_volume() -> None:
    compose = _compose()
    services = compose["services"]
    agent_a = services["agent-a"]
    agent_b = services["agent-b"]

    assert set(agent_a["networks"]).isdisjoint(agent_b["networks"])
    assert compose["networks"]["agent_a_relay"]["internal"] is True
    assert compose["networks"]["agent_b_relay"]["internal"] is True
    for agent in (agent_a, agent_b):
        assert "network_mode" not in agent
        assert "links" not in agent
        assert "external_links" not in agent
        assert "extra_hosts" not in agent
        assert "ports" not in agent

    a_volumes = set(agent_a["volumes"])
    b_volumes = set(agent_b["volumes"])
    assert a_volumes.isdisjoint(b_volumes)
    assert all("relay" not in volume for volume in a_volumes | b_volumes)


def test_relay_receives_no_team_identity_secret() -> None:
    services = _compose()["services"]

    assert services["agent-a"]["secrets"] == ["team_identity"]
    assert services["agent-b"]["secrets"] == ["team_identity"]
    assert "secrets" not in services["relay"]


def test_real_age_ciphertext_round_trips_and_excludes_plaintext() -> None:
    from demo.encrypted_sync.crypto import (
        decrypt_archive,
        encrypt_archive,
        generate_team_identity,
    )

    identity, recipient = generate_team_identity()
    plaintext = b"synthetic durable fact: cedar relay remembers blue"
    ciphertext = encrypt_archive(plaintext, recipient)

    assert b"age-encryption.org/v1" in ciphertext
    assert plaintext not in ciphertext
    assert decrypt_archive(ciphertext, identity) == plaintext


def test_wrong_age_identity_cannot_decrypt() -> None:
    from demo.encrypted_sync.crypto import (
        decrypt_archive,
        encrypt_archive,
        generate_team_identity,
    )

    _identity, recipient = generate_team_identity()
    wrong_identity, _wrong_recipient = generate_team_identity()
    ciphertext = encrypt_archive(b"synthetic private archive", recipient)

    with pytest.raises(ValueError, match="decrypt"):
        decrypt_archive(ciphertext, wrong_identity)


def test_relay_store_persists_only_opaque_bytes_and_orders_by_logical_clock(
    tmp_path: Path,
) -> None:
    from demo.encrypted_sync.relay import RelayStore

    store = RelayStore(tmp_path)
    older = b"age-encryption.org/v1\nolder opaque bytes"
    newer = b"age-encryption.org/v1\nnewer opaque bytes"

    older_receipt = store.put(older, logical_clock=4)
    newer_receipt = store.put(newer, logical_clock=5)

    assert older_receipt.object_id != newer_receipt.object_id
    assert store.latest().payload == newer
    assert store.latest().logical_clock == 5
    assert list(tmp_path.rglob("*.synapt-archive")) == []
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_relay_store_refuses_clock_rollback(tmp_path: Path) -> None:
    from demo.encrypted_sync.relay import RelayStore

    store = RelayStore(tmp_path)
    store.put(b"first", logical_clock=7)

    with pytest.raises(ValueError, match="logical clock"):
        store.put(b"stale", logical_clock=6)


def test_push_exports_then_encrypts_before_upload(monkeypatch, tmp_path: Path) -> None:
    from demo.encrypted_sync import sync

    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "export.synapt-archive"
    archive.write_bytes(b"plain portable archive")
    uploaded: dict[str, object] = {}

    def fake_export(project_dir: Path, output_path: Path, **_kwargs):
        assert project_dir == project
        assert output_path == archive
        return archive, {"version": "1"}

    def fake_encrypt(payload: bytes, recipient: str) -> bytes:
        assert payload == b"plain portable archive"
        assert recipient == "age1synthetic"
        return b"opaque age payload"

    def fake_upload(relay_url: str, payload: bytes, logical_clock: int):
        uploaded.update(url=relay_url, payload=payload, clock=logical_clock)
        return {"object_id": "obj-1", "logical_clock": logical_clock}

    monkeypatch.setattr(sync, "export_recall_archive", fake_export)
    monkeypatch.setattr(sync, "encrypt_archive", fake_encrypt)
    monkeypatch.setattr(sync, "upload_ciphertext", fake_upload)

    receipt = sync.push_project_archive(
        project,
        relay_url="http://relay:8080",
        recipient="age1synthetic",
        logical_clock=8,
        archive_path=archive,
    )

    assert uploaded == {
        "url": "http://relay:8080",
        "payload": b"opaque age payload",
        "clock": 8,
    }
    assert receipt.object_id == "obj-1"


def test_pull_decrypts_then_merge_imports_the_real_archive(
    monkeypatch, tmp_path: Path
) -> None:
    from demo.encrypted_sync import sync

    project = tmp_path / "project"
    project.mkdir()
    imported: dict[str, object] = {}

    monkeypatch.setattr(
        sync,
        "download_latest_ciphertext",
        lambda _url: (b"opaque", {"object_id": "obj-2", "logical_clock": 9}),
    )
    monkeypatch.setattr(
        sync,
        "decrypt_archive",
        lambda payload, identity: b"portable archive"
        if (payload, identity) == (b"opaque", "AGE-SECRET-KEY-SYNTHETIC")
        else b"wrong",
    )

    def fake_import(project_dir: Path, archive_path: Path, *, mode: str):
        imported.update(
            project=project_dir,
            bytes=archive_path.read_bytes(),
            mode=mode,
        )
        return {"knowledge_count": 1}

    monkeypatch.setattr(sync, "import_recall_archive", fake_import)

    receipt = sync.pull_project_archive(
        project,
        relay_url="http://relay:8080",
        identity="AGE-SECRET-KEY-SYNTHETIC",
    )

    assert imported == {
        "project": project,
        "bytes": b"portable archive",
        "mode": "merge",
    }
    assert receipt.object_id == "obj-2"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("query_positive_control", False),
        ("target_absent_before_save", False),
        ("target_present_after_sync", False),
        ("plaintext_probe_positive_control", False),
        ("relay_plaintext_hits", 1),
        ("relay_object_files", 0),
        ("relay_has_team_identity", True),
        ("local_read_after_relay_stop", False),
        ("direct_agent_route", True),
    ],
)
def test_demo_refuses_each_missing_acceptance_fruit(field: str, bad_value: object) -> None:
    from demo.encrypted_sync.run_demo import DemoFailure, validate_evidence

    evidence: dict[str, object] = {
        "query_positive_control": True,
        "target_absent_before_save": True,
        "target_present_after_sync": True,
        "plaintext_probe_positive_control": True,
        "relay_plaintext_hits": 0,
        "relay_object_files": 2,
        "relay_has_team_identity": False,
        "local_read_after_relay_stop": True,
        "direct_agent_route": False,
    }
    evidence[field] = bad_value

    with pytest.raises(DemoFailure, match=field):
        validate_evidence(evidence)


def test_real_local_composition_transfers_then_survives_relay_shutdown(
    monkeypatch, tmp_path: Path
) -> None:
    """Run the real archive, age, HTTP relay, merge, and query path together."""
    from demo.encrypted_sync.crypto import generate_team_identity
    from demo.encrypted_sync.relay import RelayHTTPServer, RelayStore
    from demo.encrypted_sync.sync import pull_project_archive, push_project_archive
    from synapt.recall.server import recall_quick, recall_save

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    relay_root = tmp_path / "relay"
    source.mkdir()
    destination.mkdir()
    target = "synthetic team fact: juniper carries violet memory"
    control = "synthetic local control: bronze lookup is operational"
    monkeypatch.setattr("synapt.recall.server.get_embedding_provider", lambda: None)

    monkeypatch.chdir(destination)
    assert "Knowledge node saved:" in recall_save(control, category="decision")
    assert control in recall_quick(control)
    assert target not in recall_quick(target)

    monkeypatch.chdir(source)
    assert "Knowledge node saved:" in recall_save(target, category="decision")

    identity, recipient = generate_team_identity()
    server = RelayHTTPServer(("127.0.0.1", 0), RelayStore(relay_root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    relay_url = f"http://127.0.0.1:{server.server_port}"
    try:
        push_project_archive(
            source,
            relay_url=relay_url,
            recipient=recipient,
            logical_clock=1,
        )
        assert all(target.encode() not in path.read_bytes() for path in relay_root.iterdir())

        pull_project_archive(destination, relay_url=relay_url, identity=identity)
        monkeypatch.chdir(destination)
        assert target in recall_quick(target)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    # The network path is gone. The imported memory remains locally readable.
    monkeypatch.chdir(destination)
    assert target in recall_quick(target)


@pytest.mark.skipif(
    os.environ.get("SYNAPT_RUN_ENCRYPTED_SYNC_DOCKER") != "1",
    reason="set SYNAPT_RUN_ENCRYPTED_SYNC_DOCKER=1 to build and run the Docker spike",
)
def test_docker_spike_proves_transfer_opacity_and_local_first() -> None:
    proc = subprocess.run(
        [sys.executable, str(SPIKE_ROOT / "run_demo.py"), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    evidence = json.loads(proc.stdout)

    assert evidence["query_positive_control"] is True
    assert evidence["target_absent_before_save"] is True
    assert evidence["target_present_after_sync"] is True
    assert evidence["plaintext_probe_positive_control"] is True
    assert evidence["relay_plaintext_hits"] == 0
    assert evidence["relay_object_files"] > 0
    assert evidence["relay_has_team_identity"] is False
    assert evidence["local_read_after_relay_stop"] is True
    assert evidence["direct_agent_route"] is False
    assert evidence["encryption"] == "age-x25519"
    assert evidence["merge"] == "relay last-write-wins; recall archive merge-import"
