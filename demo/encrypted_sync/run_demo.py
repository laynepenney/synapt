"""Run the reproducible encrypted-sync Spike A evidence sequence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


SPIKE_ROOT = Path(__file__).resolve().parent
COMPOSE = SPIKE_ROOT / "compose.yaml"
TARGET_FACT = "synthetic team fact: the cedar relay carries cobalt memory"
CONTROL_FACT = "synthetic local control: amber query path is operational"


class DemoFailure(RuntimeError):
    pass


def validate_evidence(evidence: dict[str, object]) -> None:
    """Refuse a successful demo exit when any acceptance fruit is missing."""
    required = {
        "query_positive_control": True,
        "target_absent_before_save": True,
        "target_present_after_sync": True,
        "plaintext_probe_positive_control": True,
        "relay_plaintext_hits": 0,
        "relay_has_team_identity": False,
        "local_read_after_relay_stop": True,
        "direct_agent_route": False,
    }
    failures = [
        f"{name}={evidence.get(name)!r}, expected {expected!r}"
        for name, expected in required.items()
        if evidence.get(name) != expected
    ]
    if not isinstance(evidence.get("relay_object_files"), int) or int(
        evidence["relay_object_files"]
    ) < 1:
        failures.append("relay_object_files must be at least one")
    if failures:
        raise DemoFailure("encrypted-sync evidence failed: " + "; ".join(failures))


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            command,
            cwd=SPIKE_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DemoFailure(f"command timed out: {' '.join(command)}") from exc
    if check and proc.returncode != 0:
        raise DemoFailure(
            f"command failed ({proc.returncode}): {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def _compose_command(project_name: str, *args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        project_name,
        "-f",
        str(COMPOSE),
        *args,
    ]


def _compose_exec(
    project_name: str,
    service: str,
    *args: str,
    env: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(
        _compose_command(project_name, "exec", "-T", service, *args),
        env=env,
        check=check,
    )


def _agent(
    project_name: str,
    service: str,
    *args: str,
    env: dict[str, str],
) -> dict:
    proc = _compose_exec(
        project_name,
        service,
        "python",
        "-m",
        "demo.encrypted_sync.agent",
        *args,
        env=env,
    )
    return json.loads(proc.stdout)


def run(*, keep: bool = False) -> dict[str, object]:
    project_name = f"synapt-encrypted-sync-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="synapt-encrypted-sync-key-") as tmp:
        identity_path = Path(tmp) / "team-identity.txt"
        identity_path.touch(mode=0o600)
        env = dict(os.environ)
        env["SYNAPT_TEAM_IDENTITY_FILE"] = str(identity_path)

        try:
            _run(_compose_command(project_name, "build"), env=env)
            key_proc = _run(
                _compose_command(
                    project_name,
                    "run",
                    "--rm",
                    "--no-deps",
                    "--entrypoint",
                    "python",
                    "agent-a",
                    "-m",
                    "demo.encrypted_sync.agent",
                    "keygen",
                ),
                env=env,
            )
            key = json.loads(key_proc.stdout)
            identity_path.write_text(key["identity"] + "\n", encoding="utf-8")
            identity_path.chmod(0o600)

            _run(_compose_command(project_name, "up", "-d"), env=env)

            _agent(project_name, "agent-b", "save", CONTROL_FACT, env=env)
            control = _agent(project_name, "agent-b", "query", CONTROL_FACT, env=env)
            before = _agent(project_name, "agent-b", "query", TARGET_FACT, env=env)

            _agent(project_name, "agent-a", "save", TARGET_FACT, env=env)
            _agent(
                project_name,
                "agent-a",
                "push",
                "--relay",
                "http://relay:8080",
                env=env,
            )
            _agent(
                project_name,
                "agent-b",
                "pull",
                "--relay",
                "http://relay:8080",
                env=env,
            )
            after = _agent(project_name, "agent-b", "query", TARGET_FACT, env=env)

            route_probe = _compose_exec(
                project_name,
                "agent-a",
                "python",
                "-c",
                (
                    "import socket,sys; "
                    "sys.exit(0 if socket.getaddrinfo('agent-b', 9) else 1)"
                ),
                env=env,
                check=False,
            )

            relay_probe = _compose_exec(
                project_name,
                "relay",
                "sh",
                "-c",
                (
                    "probe_dir=$(mktemp -d); "
                    "printf '%s\\n' \"$1\" > \"$probe_dir/control\"; "
                    "control=$(grep -R -a -F -l -- \"$1\" \"$probe_dir\" | wc -l); "
                    "hits=$(grep -R -a -F -l -- \"$1\" /relay-data 2>/dev/null | wc -l); "
                    "files=$(find /relay-data -type f | wc -l); "
                    "rm -rf \"$probe_dir\"; "
                    "printf '{\"control\":%s,\"hits\":%s,\"files\":%s}\\n' "
                    "\"$control\" \"$hits\" \"$files\""
                ),
                "probe",
                TARGET_FACT,
                env=env,
            )
            relay_evidence = json.loads(relay_probe.stdout)
            relay_secret_probe = _compose_exec(
                project_name,
                "relay",
                "test",
                "-e",
                "/run/secrets/team_identity",
                env=env,
                check=False,
            )

            _run(_compose_command(project_name, "stop", "relay"), env=env)
            offline = _agent(project_name, "agent-b", "query", TARGET_FACT, env=env)

            evidence: dict[str, object] = {
                "query_positive_control": bool(control["found"]),
                "target_absent_before_save": not bool(before["found"]),
                "target_present_after_sync": bool(after["found"]),
                "plaintext_probe_positive_control": int(relay_evidence["control"]) > 0,
                "relay_plaintext_hits": int(relay_evidence["hits"]),
                "relay_object_files": int(relay_evidence["files"]),
                "relay_has_team_identity": relay_secret_probe.returncode == 0,
                "local_read_after_relay_stop": bool(offline["found"]),
                "direct_agent_route": route_probe.returncode == 0,
                "encryption": "age-x25519",
                "merge": "relay last-write-wins; recall archive merge-import",
                "limit": (
                    "shared-kernel containers prove the data path, not container "
                    "escape resistance"
                ),
            }
            validate_evidence(evidence)
            return evidence
        finally:
            if keep:
                print(
                    f"containers retained under compose project {project_name}",
                    file=sys.stderr,
                )
            else:
                _run(
                    _compose_command(project_name, "down", "-v", "--remove-orphans"),
                    env=env,
                    check=False,
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)
    evidence = run(keep=args.keep)
    if args.json:
        print(json.dumps(evidence, sort_keys=True))
    else:
        print("Real encryption, real relay.")
        print("Conflict merge and key rotation are the next two layers.")
        for key, value in evidence.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
