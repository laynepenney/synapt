"""Runtime compaction summaries captured as indexed continuity metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

from synapt.recall.core import atomic_json_write, project_data_dir
from synapt.recall.scrub import scrub_text

SCHEMA_VERSION = 1
SUMMARY_TEXT_BYTES = 32 * 1024
SUMMARY_INDEX_LIMIT = 50
AGENT_DIRECTIVE_BYTES = 4 * 1024
_CLAUDE_PREFIX = (
    "This session is being continued from a previous conversation that ran "
    "out of context. The summary below covers the earlier portion of the "
    "conversation."
)


def is_claude_compaction_summary(text: str) -> bool:
    """Return whether text is Claude's synthetic post-compaction handoff."""
    return text.strip().startswith(_CLAUDE_PREFIX)


def _bounded_text(value: str, limit: int = SUMMARY_TEXT_BYTES) -> tuple[str, bool]:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value, False
    return raw[:limit].decode("utf-8", errors="ignore").rstrip() + "\n[summary truncated]", True


def _message_text(entry: dict) -> str:
    content = entry.get("message", {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def extract_compaction_summaries(path: Path) -> list[dict]:
    """Locate runtime-authored compaction handoffs in one indexed transcript."""
    summaries: list[dict] = []
    claude_boundary = False
    codex_session_id = path.stem

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                # Once a boundary is seen, an unreadable intervening record
                # breaks provenance. Fail closed rather than letting a later
                # user-authored imitation inherit the stale latch.
                if line.strip():
                    claude_boundary = False
                continue

            entry_type = entry.get("type")
            payload = entry.get("payload", {})
            if entry_type == "session_meta" and isinstance(payload, dict):
                codex_session_id = str(payload.get("id") or codex_session_id)
                continue

            if entry_type == "compacted":
                # Codex currently persists the generated handoff only as
                # encrypted_content. Record the boundary without claiming the
                # plaintext is observable.
                summaries.append({
                    "runtime": "codex",
                    "session_id": codex_session_id,
                    "timestamp": str(entry.get("timestamp") or ""),
                    "source_path": str(path),
                    "summary": None,
                    "status": "encrypted-unavailable",
                    "truncated": False,
                })
                continue

            if entry_type == "system" and entry.get("subtype") == "compact_boundary":
                claude_boundary = True
                continue

            if not claude_boundary or entry_type != "user":
                continue

            # A compaction boundary authorizes exactly the first subsequent
            # user-shaped record as the runtime handoff opportunity. Runtime
            # attachment/system records may intervene, but an ordinary user
            # turn consumes the latch so a later imitation cannot inherit
            # runtime provenance.
            claude_boundary = False
            text = _message_text(entry).strip()
            if not is_claude_compaction_summary(text):
                continue
            marker = "\n\nSummary:\n"
            summary = text.split(marker, 1)[1].strip() if marker in text else text[len(_CLAUDE_PREFIX):].strip()
            # Persistence must fail closed. The caller deliberately withholds
            # its no-op signature when sidecar indexing raises, so a scrubber
            # failure cannot certify or persist plaintext.
            summary = scrub_text(summary)
            summary, truncated = _bounded_text(summary)
            summaries.append({
                "runtime": "claude",
                "session_id": str(entry.get("sessionId") or path.stem),
                "timestamp": str(entry.get("timestamp") or ""),
                "source_path": str(path),
                "summary": summary or None,
                "status": "available" if summary else "unavailable",
                "truncated": truncated,
            })

    return summaries


def compaction_index_path(project: Path | None = None) -> Path:
    return project_data_dir(project) / "compaction-summaries.json"


def _current_schema(data: object) -> bool:
    return (
        isinstance(data, dict)
        and type(data.get("schema_version")) is int
        and data["schema_version"] == SCHEMA_VERSION
    )


def _summary_worktree(item: dict) -> str | None:
    worktree = item.get("worktree")
    if isinstance(worktree, str):
        worktree = worktree.strip()
        if (
            worktree
            and worktree not in {".", ".."}
            and "/" not in worktree
            and "\\" not in worktree
        ):
            return worktree
    source_path = item.get("source_path")
    if isinstance(source_path, str) and source_path:
        source = Path(source_path)
        inferred = source.parent.parent.name
        if (
            source.parent.name == "transcripts"
            and inferred
            and inferred not in {".", ".."}
        ):
            return inferred
    return None


def compaction_index_ready(project: Path | None = None) -> bool:
    """Return whether the current metadata schema is present and readable."""
    path = compaction_index_path(project)
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return False
    if not _current_schema(data):
        return False
    summaries = data.get("summaries")
    return (
        isinstance(summaries, list)
        and all(
            isinstance(item, dict) and _summary_worktree(item) is not None
            for item in summaries
        )
    )


def update_compaction_summary_index(
    source_dirs: list[Path],
    *,
    project: Path,
    previous_manifest: dict | None = None,
) -> None:
    """Refresh summaries for transcript files parsed by this index build."""
    destination = compaction_index_path(project)
    existing: list[dict] = []
    if previous_manifest and destination.exists():
        try:
            data = json.loads(destination.read_text(encoding="utf-8"))
            if (
                _current_schema(data)
                and isinstance(data.get("summaries"), list)
            ):
                existing = [
                    item for item in data["summaries"]
                    if isinstance(item, dict) and _summary_worktree(item) is not None
                ]
        except (OSError, json.JSONDecodeError, AttributeError):
            existing = []

    exact_stamps: dict[str, tuple[float, int]] = {}
    legacy_stamps: dict[tuple[str, str], tuple[float, int]] = {}
    if previous_manifest:
        for item in previous_manifest.get("source_files", []):
            if isinstance(item, dict):
                stamp = (item.get("mtime"), item.get("size"))
                source_path = item.get("source_path")
                if isinstance(source_path, str) and source_path:
                    exact_stamps[source_path] = stamp
                else:
                    legacy_stamps[(item.get("dir"), item.get("name"))] = stamp

    current_files: list[Path] = []
    changed: list[Path] = []
    for source_dir in source_dirs:
        for path in sorted(source_dir.glob("*.jsonl")):
            current_files.append(path)
            stat = path.stat()
            prior = exact_stamps.get(str(path))
            if prior is None:
                prior = legacy_stamps.get((source_dir.name, path.name))
            if prior != (stat.st_mtime, stat.st_size):
                changed.append(path)

    changed_paths = {str(path) for path in changed}
    current_paths = {str(path) for path in current_files}
    deleted_paths = set(exact_stamps) - current_paths

    def _path_worktree(path: Path) -> str:
        return path.parent.parent.name if path.parent.name == "transcripts" else project.name

    affected_worktrees = {
        str(_summary_worktree(item) or "")
        for item in existing
        if str(item.get("source_path") or "") in changed_paths | deleted_paths
    }
    paths_to_scan = set(changed)
    for path in current_files:
        if _path_worktree(path) in affected_worktrees:
            paths_to_scan.add(path)

    merged = [
        item for item in existing
        if item.get("source_path") not in changed_paths | deleted_paths
        and _summary_worktree(item) not in affected_worktrees
    ]
    for item in merged:
        # Existing records passed the resolver during loading. Persist its
        # canonical identity so padded labels and legacy source-only records
        # deduplicate with newly extracted records.
        item["worktree"] = _summary_worktree(item)
    for path in sorted(paths_to_scan):
        extracted = extract_compaction_summaries(path)
        worktree = _path_worktree(path)
        for item in extracted:
            item["worktree"] = worktree
        merged.extend(extracted)

    merged.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    # Startup consumes only the latest handoff for its own worktree. Keep one
    # per worktree so activity in a busy sibling cannot evict this worktree's
    # newest summary from the bounded shared sidecar.
    latest_by_worktree: list[dict] = []
    seen_worktrees: set[str] = set()
    for item in merged:
        worktree = _summary_worktree(item)
        if not worktree or worktree in seen_worktrees:
            continue
        item["worktree"] = worktree
        seen_worktrees.add(worktree)
        latest_by_worktree.append(item)
        if len(latest_by_worktree) >= SUMMARY_INDEX_LIMIT:
            break
    payload = {"schema_version": SCHEMA_VERSION, "summaries": latest_by_worktree}
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(payload, destination)


def latest_compaction_summary(
    project: Path | None = None,
    *,
    agent_id: str | None = None,
) -> dict | None:
    """Read the newest indexed summary without opening transcript files.

    A stable agent identity is a stricter selector than worktree identity. Old
    sidecars do not record an author, so they fail closed for an agent-scoped
    wake instead of attaching another agent's runtime handoff by CWD alone.
    """
    path = compaction_index_path(project)
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if not _current_schema(data):
        return None
    summaries = data.get("summaries", [])
    if not isinstance(summaries, list):
        return None

    from synapt.recall.core import _worktree_name
    expected_worktree = _worktree_name(project)
    for item in summaries:
        if not isinstance(item, dict):
            continue
        item_worktree = _summary_worktree(item)
        if item_worktree != expected_worktree:
            continue
        if agent_id and item.get("agent_id") != agent_id:
            continue
        return item
    return None


def latest_agent_compaction_directive(
    project: Path | None,
    agent_name: str | None,
) -> dict | None:
    """Read the durable directive explicitly addressed to one agent.

    ``.synapt/compact/<agent>.txt`` is identity-scoped at write time. Unlike
    the legacy runtime-summary sidecar, it does not infer authorship from the
    process CWD or from whichever agent happened to rebuild the shared index.
    """
    name = (agent_name or "").strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        return None
    # Runtime CWD is transcript context, not durable-store identity. A
    # configured recall root must retain authority even when the historical
    # project path moved or disappeared. Without a configured root, preserve
    # the explicit-project behavior used by isolated callers and tests.
    configured_root = os.environ.get("SYNAPT_RECALL_ROOT") or os.environ.get(
        "GRIPSPACE_ROOT"
    )
    data_dir = project_data_dir() if configured_root else project_data_dir(project)
    path = data_dir.parent / "compact" / f"{name}.txt"
    try:
        with path.open("rb") as stream:
            raw = stream.read(AGENT_DIRECTIVE_BYTES + 1)
    except OSError:
        return None
    if not raw:
        return None
    truncated = len(raw) > AGENT_DIRECTIVE_BYTES
    text = raw[:AGENT_DIRECTIVE_BYTES].decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    if truncated:
        text = text.rstrip() + "\n[directive clipped for startup]"
    return {
        "agent_name": name,
        "source_path": str(path),
        "text": text,
        "truncated": truncated,
    }


def format_agent_compaction_directive(item: dict) -> str:
    """Render identity-bound continuity separately from runtime inference."""
    return (
        "AGENT COMPACTION DIRECTIVE\n"
        f"Durable continuity addressed to {item['agent_name']}.\n"
        f"{item['text']}"
    )


def format_compaction_summary(item: dict, max_chars: int = 4000) -> str:
    """Render an explicitly-provenanced startup handoff."""
    runtime = str(item.get("runtime") or "unknown")
    timestamp = str(item.get("timestamp") or "unknown time")
    summary = item.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return (
            "LAST COMPACTION SUMMARY\n"
            f"{runtime.title()} recorded a compaction at {timestamp}, but its "
            "runtime-authored summary is not available to recall."
        )
    text = summary.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[summary clipped for SessionStart]"
    return (
        "LAST COMPACTION SUMMARY\n"
        f"Runtime-authored {runtime} handoff from {timestamp}.\n"
        f"{text}"
    )
