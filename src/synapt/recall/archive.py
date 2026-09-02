"""Transcript archiving and cloud sync for synapt recall.

Handles copying Claude Code transcripts into the project-local archive
(.synapt/recall/transcripts/) and optional HuggingFace sync for
cross-machine portability.

Data flow:
    Claude Code writes to ~/.claude/projects/<slug>/*.jsonl
    archive_transcripts() copies new files -> .synapt/recall/transcripts/
    build_index() reads from the archive (not from ~/.claude/)
    upload_to_hf() pushes new transcripts to HF dataset
    download_from_hf() pulls missing transcripts from HF dataset
"""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.core import project_data_dir, project_archive_dir

ARCHIVE_FORMAT_VERSION = "1"
ARCHIVE_EXTENSION = ".synapt-archive"
_ROOT_EXPORT_FILES = (
    "compaction-summaries.json",
    "config.json",
    "dedup_decisions.jsonl",
    "knowledge.jsonl",
    "last_sync_time",
    "reminders.json",
    "upload_manifest.json",
)


def _data_dir(project_dir: Path) -> Path:
    """Resolve the synapt recall data directory for a project (shared root)."""
    return project_data_dir(project_dir)


def _resolve_project_dir(project_dir: Path | None) -> Path:
    """Resolve the workspace root, honoring the env roots when unset.

    ``project_data_dir`` returns ``<root>/.synapt/recall``; the archive code
    wants ``<root>``. Derive it from the resolved data dir so the env overrides
    (``SYNAPT_RECALL_ROOT``, then ``GRIPSPACE_ROOT``), git-worktree, and
    gripspace inference all apply identically to the archive verbs and to every
    other recall surface -- and verify the expected suffix
    before stripping it (the same defense ``channel.py`` uses) rather than
    trusting a blind ``parents[1]``.
    """
    if project_dir is not None:
        return Path(project_dir).resolve()
    data_dir = project_data_dir(None)
    if data_dir.parent.name == ".synapt" and data_dir.name == "recall":
        return data_dir.parent.parent
    raise RuntimeError(
        f"resolved recall data dir has an unexpected shape: {data_dir} "
        f"(expected <root>/.synapt/recall); refusing to guess the root"
    )


def _index_dir(project_dir: Path) -> Path:
    """Return the recall index directory for *project_dir*."""
    return _data_dir(project_dir) / "index"


def _has_index(index_dir: Path) -> bool:
    """True when *index_dir* contains a recall DB in any supported layout."""
    return (index_dir / "recall.db").exists() or (index_dir / "index.db").exists()


def _archive_output_path(project_dir: Path, output_path: Path | None) -> Path:
    """Resolve the export destination path."""
    if output_path is not None:
        return output_path.expanduser().resolve()
    return (project_dir / f"{project_dir.name}{ARCHIVE_EXTENSION}").resolve()


def _json_dumps(obj: dict) -> bytes:
    """Serialize JSON deterministically for archive metadata."""
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


def _file_count(path: Path, pattern: str) -> int:
    """Count files under *path* matching *pattern*."""
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def _top_level_export_paths(data_dir: Path) -> list[Path]:
    """Return exportable root-level state files."""
    return [data_dir / name for name in _ROOT_EXPORT_FILES if (data_dir / name).is_file()]


def _iter_export_paths(
    data_dir: Path,
    *,
    include_transcripts: bool,
    include_channels: bool,
) -> list[tuple[Path, str]]:
    """Return archive members as ``(absolute_path, archive_name)`` pairs."""
    members: list[tuple[Path, str]] = []

    index_dir = data_dir / "index"
    if index_dir.is_dir():
        for path in sorted(p for p in index_dir.rglob("*") if p.is_file()):
            members.append((path, path.relative_to(data_dir).as_posix()))

    if include_channels:
        channels_dir = data_dir / "channels"
        if channels_dir.is_dir():
            for path in sorted(p for p in channels_dir.rglob("*") if p.is_file()):
                members.append((path, path.relative_to(data_dir).as_posix()))

    worktrees_root = data_dir / "worktrees"
    if worktrees_root.is_dir():
        for wt_dir in sorted(p for p in worktrees_root.iterdir() if p.is_dir()):
            journal_path = wt_dir / "journal.jsonl"
            if journal_path.is_file():
                members.append((journal_path, journal_path.relative_to(data_dir).as_posix()))
            transcript_dir = wt_dir / "transcripts"
            if include_transcripts and transcript_dir.is_dir():
                for path in sorted(p for p in transcript_dir.glob("*.jsonl") if p.is_file()):
                    members.append((path, path.relative_to(data_dir).as_posix()))

    for path in _top_level_export_paths(data_dir):
        members.append((path, path.relative_to(data_dir).as_posix()))

    members.sort(key=lambda item: item[1])
    return members


def _archive_manifest(
    project_dir: Path,
    *,
    include_transcripts: bool,
    include_channels: bool,
) -> dict:
    """Build manifest metadata for a portable recall archive."""
    from synapt.recall import __version__ as synapt_version
    from synapt.recall.sharded_db import ShardedRecallDB

    data_dir = _data_dir(project_dir)
    index_dir = data_dir / "index"
    worktrees_root = data_dir / "worktrees"

    chunk_count = 0
    knowledge_count = 0
    shard_count = 0
    is_sharded = False
    if _has_index(index_dir):
        db = ShardedRecallDB.open(index_dir)
        try:
            chunk_count = db.chunk_count()
            knowledge_count = len(db.load_knowledge_nodes(status=None))
            shard_count = db.shard_count
            is_sharded = not db.is_monolithic
        finally:
            db.close()

    transcripts_count = 0
    journals_count = 0
    worktree_names: list[str] = []
    if worktrees_root.is_dir():
        for wt_dir in sorted(p for p in worktrees_root.iterdir() if p.is_dir()):
            worktree_names.append(wt_dir.name)
            if (wt_dir / "journal.jsonl").exists():
                journals_count += 1
            if include_transcripts:
                transcripts_count += _file_count(wt_dir / "transcripts", "*.jsonl")

    channel_files = 0
    if include_channels:
        channel_files = _file_count(data_dir / "channels", "*.jsonl")

    return {
        "version": ARCHIVE_FORMAT_VERSION,
        "synapt_version": synapt_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # The RESOLVED data directory the bytes actually came from, recorded
        # after git-worktree and gripspace inference -- not the caller's
        # argument. Reported by the CLI and MCP surfaces so a wrong-store
        # export cannot hide behind plausible-looking counts, and so an
        # explicit --path that inference redirects cannot hide behind a
        # plausible-looking path either (recall#963 naming contract).
        "data_dir": str(data_dir),
        "source_project": project_dir.name,
        "chunk_count": chunk_count,
        "knowledge_count": knowledge_count,
        "shard_count": shard_count,
        "is_sharded": is_sharded,
        "worktrees": worktree_names,
        "worktree_count": len(worktree_names),
        "transcript_count": transcripts_count,
        "journal_count": journals_count,
        "channel_file_count": channel_files,
        "includes_transcripts": include_transcripts,
        "includes_channels": include_channels,
    }


def _tar_add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    """Add a bytes payload as a deterministic tar member."""
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    tf.addfile(info, io.BytesIO(data))


def _tar_add_file(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    """Add a file to the tar with deterministic metadata."""
    info = tarfile.TarInfo(arcname)
    stat = path.stat()
    info.size = stat.st_size
    info.mode = 0o644
    info.mtime = 0
    with open(path, "rb") as f:
        tf.addfile(info, f)


def export_recall_archive(
    project_dir: Path | None = None,
    output_path: Path | None = None,
    *,
    exclude_transcripts: bool = False,
    exclude_channels: bool = False,
) -> tuple[Path, dict]:
    """Export portable recall state to a ``.synapt-archive`` tar.gz file.

    *project_dir* is the workspace root to export. Pass ``None`` to resolve it
    the same way every other recall surface does -- ``SYNAPT_RECALL_ROOT``
    first, then ``GRIPSPACE_ROOT``, then git/gripspace inference, then the
    current directory ($HOME is refused, never used). Callers should only pass an
    explicit root when they genuinely know it: ``project_data_dir`` consults
    the env override ONLY when no root is passed, so a caller that forwards
    ``Path.cwd()`` does not merely skip the override, it suppresses it. That
    is how ``recall export`` silently exported the wrong store.
    """
    project_dir = _resolve_project_dir(project_dir)
    data_dir = _data_dir(project_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"No recall data found at {data_dir}")

    output_path = _archive_output_path(project_dir, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    include_transcripts = not exclude_transcripts
    include_channels = not exclude_channels
    manifest = _archive_manifest(
        project_dir,
        include_transcripts=include_transcripts,
        include_channels=include_channels,
    )
    members = _iter_export_paths(
        data_dir,
        include_transcripts=include_transcripts,
        include_channels=include_channels,
    )

    with tarfile.open(output_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        _tar_add_bytes(tf, "manifest.json", _json_dumps(manifest))
        for path, arcname in members:
            _tar_add_file(tf, path, arcname)

    return output_path, manifest


def _load_archive_manifest(archive_path: Path) -> dict:
    """Read and validate the manifest from a recall archive."""
    with tarfile.open(archive_path, "r:*") as tf:
        member = tf.getmember("manifest.json")
        manifest_file = tf.extractfile(member)
        if manifest_file is None:
            raise ValueError("Archive manifest is missing")
        manifest = json.loads(manifest_file.read().decode("utf-8"))
    version = str(manifest.get("version", ""))
    if version != ARCHIVE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported archive version {version!r}; expected {ARCHIVE_FORMAT_VERSION!r}"
        )
    return manifest


def _safe_extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract archive members under *dest_dir* with path traversal protection."""
    root = dest_dir.resolve()
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            if member.name == "manifest.json" or not member.isfile():
                continue
            target = (root / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"Unsafe archive member path: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            with open(target, "wb") as f:
                shutil.copyfileobj(extracted, f)


def _worktree_name_map(extracted_dir: Path, project_dir: Path) -> dict[str, str]:
    """Map archived worktree names onto destination worktree names.

    If the archive contains exactly one worktree and it differs from the
    current worktree name, remap it to the current worktree so imported
    transcripts/journal are immediately visible from the destination.
    Multi-worktree archives preserve names verbatim.
    """
    from synapt.recall.core import _worktree_name

    worktrees_root = extracted_dir / "worktrees"
    if not worktrees_root.is_dir():
        return {}

    archived = sorted(p.name for p in worktrees_root.iterdir() if p.is_dir())
    current = _worktree_name(project_dir)
    if len(archived) == 1 and archived[0] != current:
        return {archived[0]: current}
    return {name: name for name in archived}


def _copytree_contents(src: Path, dst: Path) -> None:
    """Copy all contents from *src* into *dst*."""
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _merge_transcript_archives(src_dir: Path, dst_dir: Path) -> int:
    """Merge transcript archive files with size-based dedup semantics."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    merged = 0
    for src_file in sorted(src_dir.glob("*.jsonl")):
        dst_file = dst_dir / src_file.name
        src_size = src_file.stat().st_size
        if dst_file.exists():
            dst_size = dst_file.stat().st_size
            if src_size == dst_size or src_size < dst_size:
                continue
        shutil.copy2(src_file, dst_file)
        merged += 1
    return merged


def _merge_jsonl_lines(src_path: Path, dst_path: Path) -> int:
    """Append unique raw JSONL lines from *src_path* into *dst_path*."""
    if not src_path.exists():
        return 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if dst_path.exists():
        with open(dst_path, encoding="utf-8") as f:
            existing = {line.rstrip("\n") for line in f if line.strip()}

    added = 0
    with open(dst_path, "a", encoding="utf-8") as out:
        with open(src_path, encoding="utf-8") as src:
            for line in src:
                raw = line.rstrip("\n")
                if not raw or raw in existing:
                    continue
                out.write(raw + "\n")
                existing.add(raw)
                added += 1
    return added


def _merge_reminders(src_path: Path, dst_path: Path) -> None:
    """Merge reminder JSON arrays by reminder ID."""
    try:
        src_items = json.loads(src_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    dst_items = []
    if dst_path.exists():
        try:
            dst_items = json.loads(dst_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dst_items = []

    merged: dict[str, dict] = {}
    for item in dst_items:
        if isinstance(item, dict) and item.get("id"):
            merged[item["id"]] = item
    for item in src_items:
        if isinstance(item, dict) and item.get("id"):
            merged[item["id"]] = item

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(
        json.dumps(list(merged.values()), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_knowledge_jsonl(nodes: list[dict], path: Path) -> None:
    """Rewrite ``knowledge.jsonl`` from deduplicated knowledge nodes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(nodes, key=lambda n: (n.get("updated_at", ""), n.get("id", "")))
    with open(path, "w", encoding="utf-8") as f:
        for node in ordered:
            f.write(json.dumps(node, sort_keys=True) + "\n")


def _merge_knowledge_nodes(local_nodes: list[dict], imported_nodes: list[dict]) -> list[dict]:
    """Merge knowledge nodes by ID, keeping the newest version."""
    merged: dict[str, dict] = {}
    for node in local_nodes:
        if node.get("id"):
            merged[node["id"]] = node
    for node in imported_nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        existing = merged.get(node_id)
        if existing is None or node.get("updated_at", "") >= existing.get("updated_at", ""):
            merged[node_id] = node
    return list(merged.values())


def _merge_chunks(local_chunks, imported_chunks):
    """Merge transcript chunks by stable chunk ID."""
    merged: dict[str, object] = {}
    for chunk in local_chunks:
        merged[chunk.id] = chunk
    for chunk in imported_chunks:
        merged.setdefault(chunk.id, chunk)
    chunks = list(merged.values())
    chunks.sort(key=lambda c: (c.timestamp, c.session_id, c.turn_index, c.id))
    return chunks


def _merge_pending_contradictions(local_items: list[dict], imported_items: list[dict]) -> list[dict]:
    """Merge pending contradictions by semantic key."""
    merged: dict[tuple, dict] = {}
    for item in local_items + imported_items:
        key = (
            item.get("old_node_id"),
            item.get("new_content"),
            item.get("category"),
            item.get("reason"),
            tuple(item.get("source_sessions", [])),
            item.get("claim_text"),
        )
        merged[key] = item
    return list(merged.values())


def _manifest_from_chunks(chunks: list, extra_manifests: list[dict]) -> dict:
    """Build merged manifest metadata from transcript chunks."""
    sessions: dict[str, dict] = {}
    for chunk in chunks:
        entry = sessions.setdefault(
            chunk.session_id,
            {"chunk_count": 0, "min_ts": chunk.timestamp or "", "max_ts": chunk.timestamp or ""},
        )
        entry["chunk_count"] += 1
        if chunk.timestamp:
            if not entry["min_ts"] or chunk.timestamp < entry["min_ts"]:
                entry["min_ts"] = chunk.timestamp
            if not entry["max_ts"] or chunk.timestamp > entry["max_ts"]:
                entry["max_ts"] = chunk.timestamp

    source_files: list[str] = []
    for manifest in extra_manifests:
        for path in manifest.get("source_files", []) or []:
            if path not in source_files:
                source_files.append(path)

    result = {
        "chunk_count": len(chunks),
        "session_count": len(sessions),
        "sessions": sessions,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if source_files:
        result["source_files"] = source_files
    return result


def _load_index_state(index_dir: Path) -> tuple[list, dict, list[dict], list[dict]]:
    """Load chunks, manifest, knowledge, and pending contradictions from *index_dir*."""
    if not _has_index(index_dir):
        return [], {}, [], []

    from synapt.recall.core import TranscriptIndex

    index = TranscriptIndex.load(index_dir, use_embeddings=False)
    try:
        db = index._db
        manifest = db.load_manifest() if db else {}
        knowledge = db.load_knowledge_nodes(status=None) if db else []
        pending = db.list_pending_contradictions() if db else []
        return index.chunks, manifest, knowledge, pending
    finally:
        if index._db is not None:
            index._db.close()


def _rebuild_merged_index(
    target_index_dir: Path,
    local_index_dir: Path,
    imported_index_dir: Path,
) -> dict:
    """Merge two indexes semantically and rewrite a clean destination index."""
    from synapt.recall.core import TranscriptIndex
    from synapt.recall.storage import RecallDB

    local_chunks, local_manifest, local_nodes, local_pending = _load_index_state(local_index_dir)
    imported_chunks, imported_manifest, imported_nodes, imported_pending = _load_index_state(imported_index_dir)

    merged_chunks = _merge_chunks(local_chunks, imported_chunks)
    merged_nodes = _merge_knowledge_nodes(local_nodes, imported_nodes)
    merged_pending = _merge_pending_contradictions(local_pending, imported_pending)
    merged_manifest = _manifest_from_chunks(
        merged_chunks,
        [local_manifest, imported_manifest],
    )

    with tempfile.TemporaryDirectory(prefix="synapt-recall-merge-") as tmp:
        tmp_index = Path(tmp) / "index"
        tmp_index.mkdir(parents=True, exist_ok=True)
        db = RecallDB(tmp_index / "recall.db")
        try:
            index = TranscriptIndex(
                merged_chunks,
                use_embeddings=False,
                cache_dir=tmp_index,
                db=db,
            )
            index.save(tmp_index)
            db.save_knowledge_nodes(merged_nodes)
            for item in merged_pending:
                db.add_pending_contradiction(
                    old_node_id=item.get("old_node_id"),
                    new_content=item.get("new_content", ""),
                    category=item.get("category", ""),
                    reason=item.get("reason", ""),
                    source_sessions=item.get("source_sessions", []),
                    detected_by=item.get("detected_by", "archive-import"),
                    claim_text=item.get("claim_text"),
                )
            db.save_manifest(merged_manifest)
        finally:
            db.close()

        if target_index_dir.exists():
            shutil.rmtree(target_index_dir)
        shutil.copytree(tmp_index, target_index_dir)

    return {
        "chunk_count": len(merged_chunks),
        "knowledge_count": len(merged_nodes),
        "pending_contradiction_count": len(merged_pending),
    }


def import_recall_archive(
    project_dir: Path | None,
    archive_path: Path,
    *,
    mode: str = "replace",
) -> dict:
    """Import a portable recall archive into *project_dir*.

    ``mode="replace"`` fully restores the archived data directory.
    ``mode="merge"`` merges transcripts, journals, channels, reminders,
    and reconstructs a merged monolithic recall index from both sources.

    *project_dir* follows the same rule as :func:`export_recall_archive`:
    ``None`` resolves via ``SYNAPT_RECALL_ROOT`` / ``GRIPSPACE_ROOT`` and
    inference; an explicit root suppresses the override, so pass one only when
    you truly know it.

    The returned summary's ``data_dir`` is the RESOLVED destination store
    this import wrote into, recorded after inference. The archive's own
    origin store (its manifest ``data_dir``) is carried as
    ``source_data_dir`` so the two are never confused: one names where the
    bytes came from, the other where they landed. Archives written before
    the manifest carried ``data_dir`` have no provenance to report, and the
    key is omitted rather than set to ``None``, so absent reads as absent.
    """
    project_dir = _resolve_project_dir(project_dir)
    if mode not in {"replace", "merge"}:
        raise ValueError("mode must be 'replace' or 'merge'")

    project_dir = project_dir.resolve()
    archive_path = archive_path.expanduser().resolve()
    manifest = _load_archive_manifest(archive_path)

    with tempfile.TemporaryDirectory(prefix="synapt-recall-import-") as tmp:
        extracted_dir = Path(tmp) / "archive"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_archive(archive_path, extracted_dir)

        data_dir = _data_dir(project_dir)
        if mode == "replace":
            wt_map = _worktree_name_map(extracted_dir, project_dir)
            for src_name, dst_name in wt_map.items():
                if src_name == dst_name:
                    continue
                src_wt = extracted_dir / "worktrees" / src_name
                dst_wt = extracted_dir / "worktrees" / dst_name
                if src_wt.exists() and not dst_wt.exists():
                    src_wt.rename(dst_wt)

            if data_dir.exists():
                shutil.rmtree(data_dir)
            data_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(extracted_dir, data_dir)
            summary = dict(manifest)
            source_store = summary.pop("data_dir", None)
            if source_store is not None:
                summary["source_data_dir"] = source_store
            summary["data_dir"] = str(data_dir)
            summary["mode"] = "replace"
            return summary

        # Merge mode
        data_dir.mkdir(parents=True, exist_ok=True)

        # Merge per-worktree transcripts and journals
        imported_worktrees = extracted_dir / "worktrees"
        if imported_worktrees.is_dir():
            from synapt.recall.journal import compact_journal

            wt_map = _worktree_name_map(extracted_dir, project_dir)
            for wt_dir in sorted(p for p in imported_worktrees.iterdir() if p.is_dir()):
                dst_wt_name = wt_map.get(wt_dir.name, wt_dir.name)
                dst_wt = data_dir / "worktrees" / dst_wt_name
                src_transcripts = wt_dir / "transcripts"
                if src_transcripts.is_dir():
                    _merge_transcript_archives(src_transcripts, dst_wt / "transcripts")
                src_journal = wt_dir / "journal.jsonl"
                if src_journal.is_file():
                    _merge_jsonl_lines(src_journal, dst_wt / "journal.jsonl")
                    compact_journal(dst_wt / "journal.jsonl")

        # Merge channels by appending unique JSONL lines. Leave local DB/cursors intact.
        imported_channels = extracted_dir / "channels"
        if imported_channels.is_dir():
            dst_channels = data_dir / "channels"
            for src_file in sorted(imported_channels.glob("*.jsonl")):
                _merge_jsonl_lines(src_file, dst_channels / src_file.name)

        # Merge small root-level state files.
        imported_reminders = extracted_dir / "reminders.json"
        if imported_reminders.is_file():
            _merge_reminders(imported_reminders, data_dir / "reminders.json")

        imported_dedup = extracted_dir / "dedup_decisions.jsonl"
        if imported_dedup.is_file():
            _merge_jsonl_lines(imported_dedup, data_dir / "dedup_decisions.jsonl")

        for name in (
            "compaction-summaries.json",
            "config.json",
            "upload_manifest.json",
            "last_sync_time",
        ):
            src = extracted_dir / name
            dst = data_dir / name
            if src.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        merged_index_stats = _rebuild_merged_index(
            data_dir / "index",
            _index_dir(project_dir),
            extracted_dir / "index",
        )

        # Regenerate knowledge.jsonl from merged DB state if we have one.
        merged_chunks, _manifest, merged_nodes, _pending = _load_index_state(data_dir / "index")
        if merged_nodes:
            _write_knowledge_jsonl(merged_nodes, data_dir / "knowledge.jsonl")

        summary = dict(manifest)
        summary.update(merged_index_stats)
        source_store = summary.pop("data_dir", None)
        if source_store is not None:
            summary["source_data_dir"] = source_store
        summary["data_dir"] = str(data_dir)
        summary["mode"] = "merge"
        summary["session_count"] = len({c.session_id for c in merged_chunks})
        return summary


# ---------------------------------------------------------------------------
# Sync configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "sync": {
        "provider": None,  # "hf" when configured
        "repo_id": None,
        "auto_sync": False,
        "extra_files": [],  # project-relative paths to upload alongside transcripts
    }
}


def load_sync_config(project_dir: Path) -> dict:
    """Load sync config from .synapt/recall/config.json, or return defaults."""
    config_path = _data_dir(project_dir) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return copy.deepcopy(DEFAULT_CONFIG)


def save_sync_config(project_dir: Path, config: dict) -> Path:
    """Save sync config to .synapt/recall/config.json (atomic write)."""
    from synapt.recall.core import atomic_json_write

    config_path = _data_dir(project_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(config, config_path)
    return config_path


# ---------------------------------------------------------------------------
# Local archiving
# ---------------------------------------------------------------------------


def archive_transcripts(project_dir: Path, source_dir: Path) -> list[Path]:
    """Copy new transcript files from Claude Code's source dir to the project archive.

    Skips files that already exist with the same size AND no newer mtime.
    Overwrites if the source grew since last archive, or if it was modified
    without changing length. Preserves larger archives when the source
    shrinks (e.g., /clear truncated the transcript).

    Args:
        project_dir: Root of the project (where .synapt/recall/ lives).
        source_dir: Claude Code transcript dir (~/.claude/projects/<slug>/).

    Returns:
        List of newly copied file paths in the archive.
    """
    archive_dir = project_archive_dir(project_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src_file in sorted(source_dir.glob("*.jsonl")):
        dst_file = archive_dir / src_file.name
        src_stat = src_file.stat()
        src_size = src_stat.st_size
        if dst_file.exists():
            dst_stat = dst_file.stat()
            dst_size = dst_stat.st_size
            if src_size == dst_size:
                # Equal size is NOT equal content. An edit that preserves the
                # byte count leaves the size identical, so size alone can never
                # observe it and the file is skipped on every subsequent build,
                # permanently. mtime is the second, independent signal; compare
                # it in ns so float rounding cannot make an unchanged file look
                # newer than itself. copy2 replicates the source mtime as
                # faithfully as the DESTINATION filesystem allows, which is
                # exactly where this can degrade: the archive may sit on a
                # different filesystem than the source, and one storing a
                # coarser mtime truncates the copied stamp, so the destination
                # reads as older than its own source and the file re-copies on
                # every build.
                #
                # That degeneration is accepted deliberately, because the two
                # ways of being wrong are not symmetric. A strict comparison
                # errs toward WORK: the cost is a bounded, repeated copy. A
                # tolerance would err toward SKIP, silently treating an edit
                # inside the tolerance window as unchanged -- which is the
                # exact defect class this check exists to kill.
                if src_stat.st_mtime_ns <= dst_stat.st_mtime_ns:
                    continue
            elif src_size < dst_size:
                continue  # Source shrunk (e.g., /clear truncated) — keep larger archive
        shutil.copy2(src_file, dst_file)
        copied.append(dst_file)

    return copied


# ---------------------------------------------------------------------------
# HuggingFace sync
# ---------------------------------------------------------------------------


def _get_hf_token(project_dir: Path | None = None) -> str | None:
    """Get HuggingFace token from env var, then .env in project dir or cwd."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    search_dirs = []
    if project_dir:
        search_dirs.append(project_dir)
    search_dirs.append(Path.cwd())

    for d in search_dirs:
        env_path = d / ".env"
        if env_path.exists():
            try:
                for line in env_path.read_text().splitlines():
                    if line.startswith("HF_TOKEN="):
                        return line.split("=", 1)[1].strip()
            except OSError:
                pass
    return None


def upload_to_hf(
    project_dir: Path,
    repo_id: str,
    token: str | None = None,
    extra_files: list[str] | None = None,
) -> int:
    """Upload new transcript archives to HuggingFace dataset.

    Compares local archive files against what's already on HF, uploads only
    new or changed files. Non-fatal — returns count of uploaded files.

    Args:
        project_dir: Project root containing .synapt/recall/transcripts/.
        repo_id: HuggingFace dataset repo ID (e.g., "user/synapt-session-logs").
        token: HF API token. Falls back to _get_hf_token().
        extra_files: Project-relative paths to upload alongside transcripts
            (e.g., ["docs/audit.jsonl"]).  Uploaded unscrubbed.

    Returns:
        Number of files uploaded.
    """
    token = token or _get_hf_token(project_dir)
    if not token:
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return 0

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception:
        pass

    # --- Transcript upload from all worktrees ---
    from synapt.recall.core import all_worktree_archive_dirs

    # Collect (worktree_name, file) pairs for namespaced upload
    local_entries: list[tuple[str, Path]] = []
    for archive_dir in all_worktree_archive_dirs(project_dir):
        # archive_dir is <main>/.synapt/recall/worktrees/<name>/transcripts/
        wt_name = archive_dir.parent.name
        for f in sorted(archive_dir.glob("*.jsonl")):
            local_entries.append((wt_name, f))

    # Track uploads by worktree-qualified key: "<wt_name>/<filename>" -> size
    manifest_path = _data_dir(project_dir) / "upload_manifest.json"
    uploaded_sizes: dict[str, int] = {}
    try:
        if manifest_path.exists():
            uploaded_sizes = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        pass

    # Scrub secrets before uploading to HF
    from synapt.recall.scrub import scrub_jsonl

    uploaded = 0
    for wt_name, local_file in local_entries:
        manifest_key = f"{wt_name}/{local_file.name}"
        local_size = local_file.stat().st_size
        if manifest_key in uploaded_sizes and local_size == uploaded_sizes[manifest_key]:
            continue
        # Backward compat: also skip if tracked under the old flat key
        if local_file.name in uploaded_sizes and local_size == uploaded_sizes[local_file.name]:
            # Migrate manifest key to namespaced format
            uploaded_sizes[manifest_key] = uploaded_sizes.pop(local_file.name)
            continue
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
            scrub_jsonl(local_file, tmp_path)
            api.upload_file(
                path_or_fileobj=str(tmp_path),
                path_in_repo=f"transcripts/{wt_name}/{local_file.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            uploaded_sizes[manifest_key] = local_size
            uploaded += 1
        except Exception:
            continue
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    # --- Extra project files (e.g., audit.jsonl) — unscrubbed, no dedup ---
    for rel_path in (extra_files or []):
        fp = project_dir / rel_path
        if not fp.exists():
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fp.name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            uploaded += 1
        except Exception:
            continue

    if uploaded > 0:
        _set_last_sync_time(project_dir)
        # Persist manifest atomically — uploaded_sizes includes both
        # previously-recorded and newly-uploaded entries.
        try:
            from synapt.recall.core import atomic_json_write
            atomic_json_write(uploaded_sizes, manifest_path)
        except OSError:
            pass

    return uploaded


# ---------------------------------------------------------------------------
# Sync debounce — avoid repeated HF uploads during active sessions
# ---------------------------------------------------------------------------

_SYNC_DEBOUNCE_FILE = "last_sync_time"


def _get_last_sync_time(project_dir: Path) -> float:
    """Read the last sync timestamp from .synapt/recall/last_sync_time."""
    ts_path = _data_dir(project_dir) / _SYNC_DEBOUNCE_FILE
    try:
        return float(ts_path.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return 0.0


def _set_last_sync_time(project_dir: Path) -> None:
    """Write the current timestamp to .synapt/recall/last_sync_time."""
    ts_path = _data_dir(project_dir) / _SYNC_DEBOUNCE_FILE
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    ts_path.write_text(str(time.time()))


def should_sync(project_dir: Path, min_interval_minutes: int = 10) -> bool:
    """Check if enough time has passed since the last sync.

    Args:
        project_dir: Project root.
        min_interval_minutes: Minimum minutes between syncs.

    Returns:
        True if a sync should proceed, False if debounced.
    """
    last = _get_last_sync_time(project_dir)
    elapsed = time.time() - last
    return elapsed >= (min_interval_minutes * 60)


def download_from_hf(project_dir: Path, repo_id: str, token: str | None = None) -> int:
    """Download missing transcript archives from HuggingFace dataset.

    Handles both old flat layout (``transcripts/<session>.jsonl``) and
    new worktree-namespaced layout (``transcripts/<wt>/<session>.jsonl``).
    Old flat files go into the current worktree's archive.  Namespaced
    files go into the matching ``worktrees/<wt>/transcripts/`` dir.

    Only downloads files not already present locally. Non-fatal.

    Returns:
        Number of files downloaded.
    """
    token = token or _get_hf_token(project_dir)
    if not token:
        return 0

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return 0

    api = HfApi(token=token)
    try:
        repo_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    except Exception:
        return 0

    remote_transcripts = [f for f in repo_files if f.startswith("transcripts/") and f.endswith(".jsonl")]

    # Build set of all locally known transcript filenames across all worktrees
    from synapt.recall.core import all_worktree_archive_dirs
    local_by_wt: dict[str, set[str]] = {}  # wt_name -> set of filenames
    for ad in all_worktree_archive_dirs(project_dir):
        wt_name = ad.parent.name
        local_by_wt[wt_name] = {f.name for f in ad.glob("*.jsonl")}

    # Current worktree's archive for flat (legacy) files
    default_archive = project_archive_dir(project_dir)
    default_wt_name = default_archive.parent.name
    local_by_wt.setdefault(default_wt_name, set())

    data_dir = _data_dir(project_dir)
    downloaded = 0
    for remote_path in remote_transcripts:
        parts = remote_path.split("/")  # ["transcripts", ...] or ["transcripts", "wt", "file"]

        if len(parts) == 3:
            # Namespaced: transcripts/<wt>/<session>.jsonl
            wt_name, filename = parts[1], parts[2]
        elif len(parts) == 2:
            # Legacy flat: transcripts/<session>.jsonl
            wt_name, filename = default_wt_name, parts[1]
        else:
            continue

        # Skip if already present in the matching worktree
        if filename in local_by_wt.get(wt_name, set()):
            continue

        # Determine target archive dir
        target_dir = data_dir / "worktrees" / wt_name / "transcripts"
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download to HF cache, then copy to our archive.  We can't use
            # local_dir because hf_hub_download recreates the repo path
            # structure (transcripts/<wt>/file.jsonl) which doesn't match
            # our worktrees/<wt>/transcripts/ layout.  HF manages cache
            # cleanup automatically.
            cached = hf_hub_download(
                repo_id, remote_path,
                repo_type="dataset",
                token=token,
            )
            shutil.copy2(cached, target_dir / filename)
            local_by_wt.setdefault(wt_name, set()).add(filename)
            downloaded += 1
        except Exception:
            continue

    return downloaded
