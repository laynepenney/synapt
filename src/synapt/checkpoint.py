"""Stdlib-only bounded SessionEnd recovery checkpoint.

This module deliberately lives outside ``synapt.recall``. Importing that
package initializes search and storage machinery, which is forbidden on the
three-second SessionEnd path even when no query is executed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from synapt.scrub import scrub_text, strip_system_artifacts

SCHEMA_VERSION = 1
TAIL_BYTES = 256 * 1024
EVENT_BYTES = 64 * 1024
TEXT_BYTES = 4 * 1024
PATH_BYTES = 1024
FILES_LIMIT = 32
CHECKPOINT_BYTES = 384 * 1024

_CLAUDE_PREFIX = (
    "This session is being continued from a previous conversation that ran "
    "out of context. The summary below covers the earlier portion of the "
    "conversation."
)
_PATCH_PATH_RE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.MULTILINE)
_FILE_RE = re.compile(
    r"(?:^|[\s\"'`])((?:/[^\s\"'`:,)]+?\.\w{1,10})|"
    r"(?:[A-Za-z]:\\[^\n\r\t\"'`]+?\.\w{1,10}))(?=[\s\"'`:,)]|$)"
)
def _scrub(text: str) -> str:
    return scrub_text(strip_system_artifacts(text))


def _bounded(value: str, limit: int) -> tuple[str, bool]:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value, False
    return raw[:limit].decode("utf-8", errors="ignore").rstrip(), True


def _string(payload: dict, key: str, default: str = "") -> str:
    value = payload.get(key, default)
    return value if isinstance(value, str) else default


def _runtime(payload: dict, transcript_path: str) -> str:
    runtime = _string(payload, "runtime").strip().lower()
    if runtime:
        return runtime
    if os.environ.get("CODEX_THREAD_ID") or "/.codex/" in transcript_path:
        return "codex"
    if os.environ.get("CLAUDE_PROJECT_DIR") or "/.claude/" in transcript_path:
        return "claude"
    return "unknown"


def _claude_user_text(entry: dict) -> str:
    if entry.get("type") != "user":
        return ""
    message = entry.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return _scrub(content)
    if not isinstance(content, list):
        return ""
    return _scrub("\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ))


def _paths_from_value(value, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"file_path", "path", "notebook_path"} and isinstance(nested, str):
                out.append(nested)
            elif key == "patch" and isinstance(nested, str):
                out.extend(_PATCH_PATH_RE.findall(nested))
            elif key in {"cmd", "command"} and isinstance(nested, str):
                out.extend(_FILE_RE.findall(nested))
            elif isinstance(nested, (dict, list)):
                _paths_from_value(nested, out)
    elif isinstance(value, list):
        for nested in value:
            _paths_from_value(nested, out)


def _claude_assistant(entry: dict) -> tuple[str, list[str]]:
    if entry.get("type") != "assistant":
        return "", []
    content = entry.get("message", {}).get("content")
    if not isinstance(content, list):
        return "", []
    texts: list[str] = []
    paths: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
            _paths_from_value(block["input"], paths)
    return _scrub("\n".join(texts)), paths


def _codex_text(payload: dict) -> tuple[str | None, str]:
    payload_type = payload.get("type", "")
    role = str(payload.get("role") or "")
    texts: list[str] = []
    content = payload.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"input_text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
    if payload_type == "user_message":
        role, texts = "user", [_string(payload, "message")]
    elif payload_type == "agent_message":
        role, texts = "assistant", [_string(payload, "message")]
    return (role if role in {"user", "assistant"} else None), _scrub("\n".join(texts))


def _record_paths(entry: dict, claude_paths: list[str]) -> list[str]:
    paths = list(claude_paths)
    payload = entry.get("payload")
    if isinstance(payload, dict) and payload.get("type") in {"function_call", "custom_tool_call"}:
        arguments = payload.get("arguments", payload.get("input"))
        if isinstance(arguments, str) and len(arguments) <= 32 * 1024:
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        _paths_from_value(arguments, paths)
    return paths


def capture_checkpoint(payload: dict) -> dict:
    """Capture one bounded checkpoint from the exact hook payload path."""
    transcript_path = _string(payload, "transcript_path")
    cwd = _string(payload, "cwd", os.getcwd()) or os.getcwd()
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "source": "session-end",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime(payload, transcript_path),
        "session_id": _string(payload, "session_id"),
        "reason": _string(payload, "reason", "other") or "other",
        "cwd": cwd,
        "transcript_path": transcript_path,
        "hook_event_name": _string(payload, "hook_event_name", "SessionEnd") or "SessionEnd",
        "last_user_text": None,
        "last_assistant_text": None,
        "files_touched": [],
        "tail_bytes_read": 0,
        "transcript_size": 0,
        "truncated": False,
        "parse_status": "unavailable",
    }
    if not transcript_path:
        return checkpoint

    malformed = False
    value_truncated = False
    try:
        with Path(transcript_path).open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            start = max(0, size - TAIL_BYTES)
            stream.seek(start)
            raw = stream.read(TAIL_BYTES)
    except OSError:
        return checkpoint

    checkpoint["transcript_size"] = size
    checkpoint["tail_bytes_read"] = len(raw)
    checkpoint["truncated"] = start > 0
    lines = raw.decode("utf-8", errors="replace").split("\n")
    if start > 0 and lines:
        lines = lines[1:]

    files: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            malformed = True
            continue
        if not isinstance(entry, dict):
            continue

        role: str | None = None
        text = _claude_user_text(entry)
        claude_paths: list[str] = []
        if text:
            role = None if text.startswith(_CLAUDE_PREFIX) else "user"
        else:
            text, claude_paths = _claude_assistant(entry)
            if text:
                role = "assistant"
            else:
                payload_obj = entry.get("payload")
                if isinstance(payload_obj, dict):
                    role, text = _codex_text(payload_obj)

        if role and text.strip():
            text, clipped = _bounded(text.strip(), TEXT_BYTES)
            value_truncated = value_truncated or clipped
            checkpoint[f"last_{role}_text"] = text
        files.extend(_record_paths(entry, claude_paths))

    seen: set[str] = set()
    bounded_files: list[str] = []
    for path in files:
        bounded_path, clipped = _bounded(path, PATH_BYTES)
        value_truncated = value_truncated or clipped
        if bounded_path and bounded_path not in seen:
            seen.add(bounded_path)
            bounded_files.append(bounded_path)
        if len(bounded_files) >= FILES_LIMIT:
            value_truncated = value_truncated or len(files) > len(bounded_files)
            break
    checkpoint["files_touched"] = bounded_files

    found = bool(checkpoint["last_user_text"] or checkpoint["last_assistant_text"])
    complete = bool(checkpoint["last_user_text"] and checkpoint["last_assistant_text"])
    checkpoint["truncated"] = bool(checkpoint["truncated"] or value_truncated)
    if not found:
        checkpoint["parse_status"] = "partial" if checkpoint["truncated"] or malformed else "unavailable"
    elif complete and not malformed:
        checkpoint["parse_status"] = "ok"
    else:
        checkpoint["parse_status"] = "partial"
    return checkpoint


def _namespace(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a bare namespace label")
    return value


def _linked_main_root(cwd: Path) -> Path | None:
    """Resolve an exact cwd's .git pointer without discovery or Git."""
    git_file = cwd / ".git"
    if not git_file.is_file():
        return None
    try:
        content = git_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not content.startswith("gitdir:"):
        return None
    gitdir = Path(content.split(":", 1)[1].strip())
    if not gitdir.is_absolute():
        gitdir = (cwd / gitdir).resolve()
    for parent in (gitdir, *gitdir.parents):
        if parent.name == ".git":
            return parent.parent
    return None


def _nearest_ancestor_with(cwd: Path, marker: str) -> Path | None:
    """Find one parent marker without directory enumeration or child work."""
    for candidate in (cwd, *cwd.parents):
        try:
            if (candidate / marker).exists():
                return candidate
        except OSError:
            continue
    return None


def _linked_gripspace_root(
    linked_griptree: Path,
    repository: Path | None,
) -> Path | None:
    """Resolve a linked member through one bounded set of git pointers."""
    candidates: list[Path] = []
    if repository is not None:
        candidates.append(repository)
    else:
        directory_count = 0
        try:
            with os.scandir(linked_griptree) as entries:
                for index, entry in enumerate(entries):
                    # Bound both the total directory work and the number of
                    # candidate repositories. Unrelated files must not consume
                    # the repository-candidate allowance.
                    if index >= 4096 or directory_count >= 256:
                        break
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            directory_count += 1
                            candidates.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            return None

    for candidate in candidates:
        main_repository = _linked_main_root(candidate)
        if main_repository:
            owner = _nearest_ancestor_with(
                main_repository, ".gitgrip/griptrees.json",
            )
            if owner:
                return owner
    return None


def checkpoint_path(cwd: Path) -> Path:
    """Resolve the established store with bounded parent-marker checks."""
    exact_cwd = cwd.expanduser().resolve()
    recall_root_value = os.environ.get("SYNAPT_RECALL_ROOT")
    gripspace_root_value = os.environ.get("GRIPSPACE_ROOT")
    root_value = recall_root_value or gripspace_root_value
    explicit_root = Path(root_value).expanduser().resolve() if root_value else None
    if recall_root_value:
        explicit_is_gripspace = bool(
            explicit_root
            and (explicit_root / ".gitgrip" / "griptrees.json").exists()
        )
    else:
        explicit_is_gripspace = bool(gripspace_root_value)
    repository = _nearest_ancestor_with(exact_cwd, ".git")
    linked_griptree = _nearest_ancestor_with(
        exact_cwd, ".gitgrip/griptree.json",
    )
    gripspace = _nearest_ancestor_with(
        exact_cwd, ".gitgrip/griptrees.json",
    )
    if linked_griptree:
        linked_owner = _linked_gripspace_root(linked_griptree, repository)
        if linked_owner:
            gripspace = linked_owner
    if explicit_root:
        root = explicit_root
    elif gripspace:
        root = gripspace
    elif repository:
        root = _linked_main_root(repository) or repository
    else:
        root = exact_cwd

    inferred_worktree = repository.name if repository else exact_cwd.name
    if linked_griptree:
        inferred_worktree = repository.name if repository else linked_griptree.name
    elif explicit_is_gripspace or (not explicit_root and gripspace):
        namespace_root = explicit_root or gripspace
        try:
            relative = exact_cwd.relative_to(namespace_root)
        except ValueError:
            pass
        else:
            if relative.parts:
                inferred_worktree = relative.parts[0]
    elif explicit_root:
        inferred_worktree = repository.name if repository else explicit_root.name
    worktree = _namespace(
        os.environ.get("SYNAPT_RECALL_WORKTREE") or inferred_worktree,
        "SYNAPT_RECALL_WORKTREE",
    )
    return root / ".synapt" / "recall" / "worktrees" / worktree / "checkpoint.json"


def _encoded(checkpoint: dict) -> bytes:
    return (json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_checkpoint(payload: dict) -> tuple[Path, dict]:
    checkpoint = capture_checkpoint(payload)
    destination = checkpoint_path(Path(checkpoint["cwd"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encoded(checkpoint)
    while len(encoded) > CHECKPOINT_BYTES and checkpoint["files_touched"]:
        checkpoint["files_touched"].pop()
        checkpoint["truncated"] = True
        encoded = _encoded(checkpoint)
    if len(encoded) > CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint exceeds {CHECKPOINT_BYTES} serialized bytes")

    fd, temp_name = tempfile.mkstemp(prefix=".checkpoint.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return destination, checkpoint


def read_checkpoint(project: Path) -> dict | None:
    path = checkpoint_path(project)
    try:
        if path.stat().st_size > CHECKPOINT_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (
        value
        if (
            isinstance(value, dict)
            and type(value.get("schema_version")) is int
            and value["schema_version"] == SCHEMA_VERSION
        )
        else None
    )


def _timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_newer_than(checkpoint: dict, authored_journal_timestamp: str | None) -> bool:
    captured = _timestamp(str(checkpoint.get("captured_at") or ""))
    authored = _timestamp(authored_journal_timestamp or "")
    return captured is not None and (authored is None or captured > authored)


def format_checkpoint(checkpoint: dict) -> str:
    user = checkpoint.get("last_user_text") or "unavailable in bounded transcript tail"
    assistant = checkpoint.get("last_assistant_text") or "unavailable in bounded transcript tail"
    files = checkpoint.get("files_touched") or []
    file_text = ", ".join(str(path) for path in files[:FILES_LIMIT]) or "none observed"
    return (
        "LAST CHECKPOINT\n"
        "Raw transcript tail captured at SessionEnd. Not an authored journal.\n"
        f"Status: {checkpoint.get('parse_status', 'unavailable')}"
        f"{' (truncated)' if checkpoint.get('truncated') else ''}\n"
        f"User: {user}\nAssistant: {assistant}\nFiles: {file_text}"
    )


def _read_event(path: str) -> dict:
    stream = sys.stdin.buffer if path == "-" else Path(path).open("rb")
    close = path != "-"
    try:
        raw = stream.read(EVENT_BYTES + 1)
    finally:
        if close:
            stream.close()
    if len(raw) > EVENT_BYTES:
        raise ValueError(f"event JSON exceeds {EVENT_BYTES} bytes")
    value = json.loads(raw.decode("utf-8")) if raw.strip() else {}
    if not isinstance(value, dict):
        raise ValueError("event JSON must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synapt recall checkpoint")
    parser.add_argument("--event-json", default="-", help="Hook event JSON path, or - for stdin")
    args = parser.parse_args(argv)
    try:
        write_checkpoint(_read_event(args.event_json))
    except Exception as exc:
        print(f"synapt recall checkpoint: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
