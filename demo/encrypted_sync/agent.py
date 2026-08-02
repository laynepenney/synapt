"""Container-side commands for the encrypted-sync demonstration."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

from synapt.recall.server import recall_quick, recall_save

from .crypto import generate_team_identity, recipient_from_identity
from .sync import pull_project_archive, push_project_archive


DEFAULT_PROJECT = Path("/workspace")
DEFAULT_IDENTITY = Path("/run/secrets/team_identity")


@contextlib.contextmanager
def _inside(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _identity(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"team identity is empty: {path}")
    return text


def _print(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _next_clock(project: Path) -> int:
    path = project / ".synapt" / "encrypted-sync-clock"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = int(path.read_text(encoding="utf-8")) if path.exists() else 0
    updated = current + 1
    path.write_text(str(updated), encoding="utf-8")
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encrypted-sync spike agent")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("keygen")

    save = subparsers.add_parser("save")
    save.add_argument("content")
    save.add_argument("--category", default="decision")

    query = subparsers.add_parser("query")
    query.add_argument("needle")

    push = subparsers.add_parser("push")
    push.add_argument("--relay", required=True)

    pull = subparsers.add_parser("pull")
    pull.add_argument("--relay", required=True)

    args = parser.parse_args(argv)
    if args.command == "keygen":
        identity, recipient = generate_team_identity()
        _print({"identity": identity, "recipient": recipient})
        return 0

    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    if args.command == "save":
        with _inside(project):
            result = recall_save(content=args.content, category=args.category)
        saved = "Knowledge node saved:" in result or "Knowledge node updated:" in result
        _print({"saved": saved, "result": result})
        return 0 if saved else 1

    if args.command == "query":
        with _inside(project):
            result = recall_quick(args.needle)
        found = args.needle.casefold() in result.casefold()
        _print({"found": found, "result": result})
        return 0

    identity = _identity(args.identity)
    if args.command == "push":
        clock = _next_clock(project)
        receipt = push_project_archive(
            project,
            relay_url=args.relay,
            recipient=recipient_from_identity(identity),
            logical_clock=clock,
        )
        _print(
            {
                "object_id": receipt.object_id,
                "logical_clock": receipt.logical_clock,
            }
        )
        return 0


    receipt = pull_project_archive(
        project,
        relay_url=args.relay,
        identity=identity,
    )
    _print(
        {
            "object_id": receipt.object_id,
            "logical_clock": receipt.logical_clock,
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
