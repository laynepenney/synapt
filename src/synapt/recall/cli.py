#!/usr/bin/env python3
"""CLI for synapt: build, search, and inspect per-project session indexes.

Usage:
    # One-time setup: build index, register MCP, install hook
    synapt setup

    # Setup with HuggingFace sync
    synapt setup --sync hf:user/dataset-name

    # Build index for current project
    synapt build

    # Build with explicit source
    synapt build --source ~/.claude/projects/-Users-me-Development-myproject

    # Build from ChatGPT export
    synapt build --chatgpt-archive ~/Downloads/chatgpt-export.zip

    # Build from HuggingFace
    HF_TOKEN=... synapt build --hf user/dataset-name

    # Search current project's index
    synapt search "quality curve" --max-chunks 5

    # Progressive search (most recent sessions first)
    synapt search "harness bug" --max-sessions 3

    # Search with date filtering
    synapt search "what errors" --after 2026-02-28 --before 2026-03-01

    # Index stats
    synapt stats

    # Sync transcripts to/from HuggingFace
    synapt sync push
    synapt sync pull
    synapt sync both

    # Install global hooks (SessionStart, SessionEnd, PreCompact)
    synapt install-hook

    # Hook-triggered rebuild (called by PreCompact hook)
    synapt rebuild
    synapt rebuild --sync
"""

from __future__ import annotations

import argparse
import importlib.resources
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from synapt.recall.core import (
    atomic_json_write,
    format_size,
    TranscriptIndex,
    build_index,
    project_data_dir,
    project_index_dir,
    project_archive_dir,
    project_slug,
    project_transcript_dir,
    project_transcript_dirs,
    all_worktree_archive_dirs,
    _is_real_user_message,
    _extract_user_text,
    _extract_assistant_content,
)
from synapt.recall.chatgpt import parse_chatgpt_archive
from synapt.recall.journal import (
    latest_transcript_path,
    extract_session_id,
    _journal_path,
    _read_all_session_ids,
    auto_extract_entry,
    append_entry,
    split_journal_field,
)

logger = logging.getLogger("synapt.recall.cli")

_WAKE_JOURNAL_ENTRY_LIMIT = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_index_dir(args: argparse.Namespace) -> Path:
    """Resolve the index directory from args or cwd."""
    if getattr(args, "index", None):
        return Path(args.index).expanduser()
    if getattr(args, "out", None):
        return Path(args.out).expanduser()
    return project_index_dir()


def _check_legacy_index() -> Path | None:
    """Check for old ~/.synapse-recall/<slug>/ index location."""
    slug = project_slug()
    legacy = Path.home() / ".synapse-recall" / slug
    if legacy.exists() and (
        (legacy / "recall.db").exists() or (legacy / "chunks.jsonl").exists()
    ):
        return legacy
    return None


def _ensure_gitignore(project_dir: Path) -> None:
    """Add .synapt/ to .gitignore if not already present."""
    gitignore_path = project_dir / ".gitignore"
    new_entry = ".synapt/"
    old_entries = [".synapse-recall/", ".synapse/"]

    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        # Migrate any old entries
        changed = False
        for old_entry in old_entries:
            if old_entry in lines and new_entry not in lines:
                lines = [new_entry if l == old_entry else l for l in lines]
                changed = True
        if changed:
            gitignore_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        if new_entry in lines:
            return
        if not content.endswith("\n"):
            content += "\n"
        content += f"{new_entry}\n"
        gitignore_path.write_text(content, encoding="utf-8")
    else:
        gitignore_path.write_text(f"{new_entry}\n", encoding="utf-8")


def _codex_home() -> Path:
    override = os.environ.get("CODEX_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".codex"


def _install_codex_skill(skill_name: str = "dev-loop") -> Path | None:
    """Install a packaged Codex skill into the user's Codex home."""
    try:
        skill_text = (
            importlib.resources.files("synapt.resources")
            .joinpath("skills", skill_name, "SKILL.md")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        logger.warning("Packaged skill %s not found; skipping Codex skill install", skill_name)
        return None

    dest = _codex_home() / "skills" / skill_name / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.read_text(encoding="utf-8") != skill_text:
        dest.write_text(skill_text, encoding="utf-8")
    return dest


def _build_lock_busy_message(data_dir: Path, name: str = "build.lock") -> str:
    """Who holds the lock, from the stamp the holder wrote on acquire."""
    try:
        stamp = (data_dir / name).read_text(encoding="utf-8").strip()
    except OSError:
        stamp = ""
    return f"held by {stamp}" if stamp else "holder unknown"


def _acquire_build_lock(data_dir: Path, timeout: float = 60.0, name: str = "build.lock") -> "int | None":
    """Acquire an exclusive file lock for index builds (or, by *name*, for any
    other single-flight job such as ``catchup``).

    Returns the lock file descriptor on success, None if the lock could not
    be acquired within *timeout* seconds (another holder is running). On
    success the holder stamps ``pid … since …`` into the file, so a waiter
    that gives up can say WHO it waited on rather than only that it waited.
    """
    import errno
    import time
    from synapt.recall._filelock import lock_exclusive_nb

    lock_path = data_dir / name
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock_exclusive_nb(fd)
            try:
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, f"pid {os.getpid()} since {datetime.now().astimezone().isoformat(timespec='seconds')}\n".encode())
            except OSError:
                pass  # the stamp is a courtesy; the lock is the guarantee
            return fd
        except OSError as exc:
            # Only retry on lock contention; other errors are fatal
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                os.close(fd)
                return None
            if time.monotonic() >= deadline:
                os.close(fd)
                return None
            time.sleep(0.5)


def _release_build_lock(fd: int) -> None:
    """Release the build file lock."""
    from synapt.recall._filelock import unlock
    try:
        unlock(fd)
    finally:
        os.close(fd)


def _build_journal_files(project_dir: Path) -> list[Path]:
    """Every journal file this build reads: local, plus each other worktree's.

    Extracted so the no-op SIGNATURE and the build itself cannot read different
    sets. Two independent derivations of "the inputs" are two chances to
    disagree, and a signature covering fewer files than the build reads produces
    a FALSE no-op -- the build reports "up to date" while a real change sits
    unindexed, which is the one failure direction with no external symptom.
    """
    from synapt.recall.journal import _journal_path

    local_journal = _journal_path(project_dir)
    files = [local_journal]
    for wt_archive in all_worktree_archive_dirs(project_dir):
        # Archive dir is <main>/.synapt/recall/worktrees/<name>/transcripts/;
        # the journal sits beside it at .../<name>/journal.jsonl
        wt_journal = wt_archive.parent / "journal.jsonl"
        if wt_journal.resolve() != local_journal.resolve() and wt_journal.exists():
            files.append(wt_journal)
    return files


def _archive_and_build(
    project_dir: Path,
    source_dirs: list[Path] | None = None,
    use_embeddings: bool = True,
    incremental: bool = False,
    chatgpt_archive: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> TranscriptIndex | None:
    """Archive transcripts and build the index. Shared by build/rebuild/setup.

    Acquires an exclusive file lock so only one process builds at a time.
    Other worktrees' builds will wait up to 60s for the lock.

    1. Archive transcripts from Claude Code source dir -> .synapt/recall/transcripts/
    2. Build index from archive (not directly from ~/.claude/)

    Returns the final TranscriptIndex, or None if no chunks found.
    """
    data_dir = project_data_dir(project_dir)
    if progress:
        progress("waiting_for_lock")
    lock_fd = _acquire_build_lock(data_dir)
    if lock_fd is None:
        if progress:
            progress("lock_timeout")
        print(f"  Warning: another build is in progress (timed out waiting for lock; {_build_lock_busy_message(data_dir)})")
        return None

    try:
        return _archive_and_build_locked(
            project_dir, source_dirs, use_embeddings, incremental, chatgpt_archive,
            progress,
        )
    finally:
        _release_build_lock(lock_fd)


def _archive_and_build_locked(
    project_dir: Path,
    source_dirs: list[Path] | None,
    use_embeddings: bool,
    incremental: bool,
    chatgpt_archive: str | None,
    progress: Callable[[str], None] | None = None,
) -> TranscriptIndex | None:
    """Inner build logic — caller must hold the build lock."""
    import time as _time

    from synapt.recall.archive import archive_transcripts
    from synapt.recall.codex import archive_codex_transcripts
    from synapt.recall.storage import RecallDB

    build_t0 = _time.monotonic()
    index_dir = project_index_dir(project_dir)
    archive_dir = project_archive_dir(project_dir)

    # Step 1: Archive transcripts from Claude Code's source dir
    if progress:
        progress("archiving")
    if not source_dirs:
        source_dirs = project_transcript_dirs(project_dir)

    if source_dirs:
        for src in source_dirs:
            copied = archive_transcripts(project_dir, src)
            if copied:
                print(f"  Archived {len(copied)} new transcript(s) from {src}")

    codex_copied = archive_codex_transcripts(project_dir)
    if codex_copied:
        print(f"  Archived {len(codex_copied)} Codex transcript(s)")

    # Step 2: Determine what to build from — aggregate all worktree archives
    build_sources: list[Path] = []
    for wt_archive in all_worktree_archive_dirs(project_dir):
        build_sources.append(wt_archive)
    # Include this worktree's archive if not already found by worktree discovery
    if archive_dir.exists() and any(archive_dir.glob("*.jsonl")):
        resolved = archive_dir.resolve()
        if resolved not in {p.resolve() for p in build_sources}:
            build_sources.append(archive_dir)

    # Fall back to direct source if no archives exist yet (first run)
    if not build_sources and source_dirs:
        build_sources = [source_dirs[0]]

    # Step 3: Open or create SQLite database (shard-aware)
    index_dir.mkdir(parents=True, exist_ok=True)
    from synapt.recall.sharding import is_sharded
    if is_sharded(index_dir):
        from synapt.recall.sharded_db import ShardedRecallDB
        db = ShardedRecallDB.open(index_dir)
        print(f"  Sharded layout: {db.shard_count} data shard(s)")
    else:
        db = RecallDB(index_dir / "recall.db")

    # Fingerprint the inputs and short-circuit a no-op run.
    #
    # NOT before all work: archiving (Step 1) has already run and scanned the
    # source dirs by this point. It copied nothing -- which is exactly why the
    # signature matches, since the signature is path, size and mtime -- but the
    # skip line must not claim a stage was skipped when it executed. An operator
    # reads that line and stops looking, so it says what actually happened.
    #
    # A fast build and a broken build look identical from outside unless the
    # build says which stages it skipped, so the no-op path REPORTS as well as
    # returning early. Computed unconditionally, not only when incremental, so a
    # full rebuild still leaves a baseline for the next incremental run --
    # otherwise the first incremental build after any full one can never be a
    # no-op and the signal looks broken.
    #
    # `is_noop` fails toward doing the work when the prior signature is absent,
    # malformed, or version-mismatched. That asymmetry is deliberate: a wrong
    # "changed" costs one unnecessary build, while a wrong "unchanged" leaves
    # real content unindexed and prints a reassuring line about it.
    from synapt.recall.build_delta import (
        compute_input_signature,
        is_noop,
        signature_from_manifest,
        signature_to_manifest,
    )
    from synapt.recall.channel import _channels_dir

    # The ChatGPT export is parsed into the index on every build, so it is a
    # build input and must be a signed one (Atlas, r2 on v3).
    archive_paths = [Path(chatgpt_archive).expanduser()] if chatgpt_archive else []

    def _readonly_signature():
        """Signature over the inputs the build only READS.

        Keyed on `build_sources` -- the ARCHIVE the build actually parses -- and
        not on `source_dirs`, the outside directories Step 1 copies FROM. Signing
        the copy source cannot see anything reaching the archive by another route
        (Codex rollouts are copied straight from the Codex sessions directory and
        never pass through source_dirs at all), and it mis-times everything else:
        a transcript written after Step 1's copy is present in source_dirs and
        absent from the archive, so it gets signed as parsed without having been.
        Sign what is PARSED, not what it was copied from (Stromus, r1 on v6).

        Journals are excluded deliberately: the build WRITES them (auto-journal
        stubs), so they differ before and after by design and cannot be used to
        detect an outside arrival. Everything here is content the build treats
        as immutable for the duration of the run, which makes a difference
        between two samples of it proof that something landed mid-build.
        """
        return compute_input_signature(
            source_dirs=build_sources,
            channels_dir=_channels_dir(project_dir),
            archive_paths=archive_paths,
        )

    readonly_before = _readonly_signature()
    build_signature = compute_input_signature(
        source_dirs=build_sources,
        channels_dir=_channels_dir(project_dir),
        journal_paths=_build_journal_files(project_dir),
        archive_paths=archive_paths,
    )
    from synapt.recall.compaction import compaction_index_ready
    if (
        incremental
        and compaction_index_ready(project_dir)
        and is_noop(signature_from_manifest(db.load_manifest()), build_signature)
    ):
        print("  Up to date: no transcript, channel or journal input changed")
        print("  Skipped: parse, enrich, index (archive scanned; nothing new)")
        return TranscriptIndex.load(index_dir)

    # Step 4: Load existing data for incremental builds
    incremental_manifest = None
    existing_chunks = []
    if incremental:
        manifest = db.load_manifest()
        if manifest.get("chunk_count"):
            incremental_manifest = manifest
            existing_chunks = db.load_chunks()
            print(f"  Incremental: {len(existing_chunks)} existing chunks")
        elif (index_dir / "manifest.json").exists():
            # Fallback: legacy JSON manifest
            try:
                with open(index_dir / "manifest.json", encoding="utf-8") as f:
                    incremental_manifest = json.load(f)
                existing_index = TranscriptIndex.load(index_dir)
                existing_chunks = existing_index.chunks
                print(f"  Incremental: {len(existing_chunks)} existing chunks")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  Warning: corrupt manifest, falling back to full rebuild ({exc})")
                incremental_manifest = None
                existing_chunks = []

    # Strip old journal chunks — they'll be re-parsed fresh below
    all_chunks = [c for c in existing_chunks if c.turn_index >= 0]

    # Detect content profile from existing chunks (if available) to set
    # sub-chunk threshold. Personal content disables sub-chunking to
    # preserve full conversation turns for retrieval (#455).
    _subchunk_min = None  # None = use default (1200)
    if all_chunks:
        from synapt.recall.content_profile import detect_content_profile, adaptive_params
        _profile = detect_content_profile(all_chunks)
        _subchunk_min = adaptive_params(_profile).subchunk_min_text
        if _subchunk_min != 1200:
            logger.info("build: content profile %s → subchunk_min_text=%d",
                         _profile.content_type, _subchunk_min)

    # Build from archived transcripts (BM25-only, no DB needed for parsing)
    if progress:
        progress("parsing")
    logger.info("build: parsing transcripts from %d source(s)...", len(build_sources))
    for build_source in build_sources:
        index = build_index(
            build_source,
            use_embeddings=False,
            incremental_manifest=incremental_manifest,
            subchunk_min_text=_subchunk_min,
        )
        all_chunks.extend(index.chunks)
    logger.info("build: parsed %d chunks in %.1fs", len(all_chunks), _time.monotonic() - build_t0)

    # ChatGPT archive (separate source)
    if chatgpt_archive:
        archive_path = Path(chatgpt_archive).expanduser()
        print(f"\n[build] Parsing ChatGPT archive {archive_path} ...")
        chatgpt_chunks = parse_chatgpt_archive(archive_path)
        print(f"[build] Parsed {len(chatgpt_chunks)} ChatGPT chunks")
        all_chunks.extend(chatgpt_chunks)

    # Channel messages → searchable chunks
    try:
        from synapt.recall.channel import _channels_dir, _read_messages, ChannelMessage
        from synapt.recall.core import TranscriptChunk
        import hashlib as _hashlib
        ch_dir = _channels_dir(project_dir)
        if ch_dir.exists():
            channel_chunks = []
            for ch_file in sorted(ch_dir.glob("*.jsonl")):
                ch_name = ch_file.stem
                messages = _read_messages(ch_file)
                # Group consecutive messages into chunks (max 10 per chunk)
                for i in range(0, len(messages), 10):
                    batch = messages[i:i+10]
                    if not batch:
                        continue
                    body_parts = []
                    for msg in batch:
                        if msg.type in ("message", "directive"):
                            body_parts.append(f"{msg.from_agent}: {msg.body}")
                    if not body_parts:
                        continue
                    text = "\n".join(body_parts)
                    seed = f"ch_{ch_name}_{batch[0].timestamp}_{i}"
                    chunk_id = f"ch_{_hashlib.sha256(seed.encode()).hexdigest()[:12]}:t{i//10}"
                    chunk = TranscriptChunk(
                        id=chunk_id,
                        session_id=f"channel_{ch_name}",
                        timestamp=batch[0].timestamp,
                        turn_index=i // 10,
                        user_text="",
                        assistant_text=text,
                    )
                    channel_chunks.append(chunk)
            if channel_chunks:
                all_chunks.extend(channel_chunks)
                print(f"  Channels: {len(channel_chunks)} chunks from {len(list(ch_dir.glob('*.jsonl')))} channel(s)")
    except Exception as exc:
        logger.debug("Channel indexing failed: %s", exc)

    # Tier 1: Synthesize auto-journal stubs for sessions without entries
    from synapt.recall.journal import _journal_path, synthesize_journal_stubs
    from synapt.recall.core import parse_journal_entries

    journal_files = _build_journal_files(project_dir)
    # Recomputed rather than taken as journal_files[0]: relying on the
    # helper's ordering would make a harmless reordering there a silent
    # bug here.
    local_journal = _journal_path(project_dir)

    transcript_chunks = [c for c in all_chunks if c.turn_index >= 0]
    if transcript_chunks:
        sessions: dict[str, list] = {}
        for c in transcript_chunks:
            sessions.setdefault(c.session_id, []).append(c)
        # Only synthesize stubs into the local journal
        synthesized = synthesize_journal_stubs(sessions, local_journal, project_root=str(project_dir))
        if synthesized:
            print(f"  Auto-journal: {synthesized} stub(s) synthesized")

    # THE JOURNAL READ BOUNDARY. Stub synthesis is done; parsing has not begun.
    # This is the journal state the index is about to reflect, and it is the
    # only sample that can tell the build's OWN writes from an outside arrival:
    # everything before this line is ours, everything after it is somebody
    # else's. v4 excluded journals from the arrival guard entirely to avoid
    # mistaking synthesis for an arrival -- correct about our writes, and it
    # surrendered every genuine journal arrival along with them (Atlas, r2 on
    # v4: an entry appended after the index save was certified as indexed and
    # the next run returned zero of it). Excluding a class to suppress a false
    # positive gives up the true positives in the same motion.
    # ONE FULL SAMPLE, TAKEN HERE, AND IT IS THE ONE THAT GETS PERSISTED.
    # v5 compared journals, then compared read-only inputs, then computed a
    # THIRD signature to store -- so the value written to the manifest was never
    # the value that was checked, and anything arriving in between was certified
    # as indexed without being read (Atlas, r2 on v5: measured, three signature
    # computations after the index write). Two samples of the same quantity are
    # not the same sample. The stored signature is therefore taken HERE, before
    # any guard runs, and the guards only decide whether to keep it: nothing that
    # happens later can enter a value that was already computed.
    signature_at_read = compute_input_signature(
        source_dirs=build_sources,
        channels_dir=_channels_dir(project_dir),
        journal_paths=journal_files,
        archive_paths=archive_paths,
    )
    readonly_at_read = _readonly_signature()

    # Journal entries → searchable chunks from ALL worktrees
    for journal_file in journal_files:
        if journal_file.exists():
            journal_chunks = parse_journal_entries(journal_file)
            if journal_chunks:
                print(f"  Journal: {len(journal_chunks)} entries from {journal_file.parent.name}")
                all_chunks.extend(journal_chunks)

    if not all_chunks:
        return None

    # Dedup by chunk id
    deduped = []
    seen_ids = set()
    for chunk in all_chunks:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            deduped.append(chunk)

    # Build final index with SQLite backend
    if progress:
        progress("indexing")
    logger.info("build: saving %d chunks to FTS5 index...", len(deduped))
    save_t0 = _time.monotonic()
    final_index = TranscriptIndex(
        deduped,
        use_embeddings=use_embeddings,
        cache_dir=index_dir,
        db=db,
    )
    final_index.save(index_dir)
    logger.info("build: FTS5 save complete in %.1fs", _time.monotonic() - save_t0)

    # Cluster chunks by topic similarity
    if progress:
        progress("clustering")
    logger.info("build: clustering %d transcript chunks...", sum(1 for c in deduped if c.turn_index >= 0))
    from synapt.recall.clustering import cluster_chunks as _cluster_chunks, generate_concat_summary
    transcript_only = [c for c in deduped if c.turn_index >= 0]
    if transcript_only:
        clusters = _cluster_chunks(transcript_only)
        if clusters:
            # Build chunk ID → TranscriptChunk lookup for summary generation
            chunk_map = {c.id: c for c in transcript_only}
            memberships = []
            for cl in clusters:
                for cid in cl["chunk_ids"]:
                    memberships.append((cl["cluster_id"], cid, cl["created_at"]))
            # Enrich each cluster with search_text from member chunk content.
            # Include user_text, tools, and files so concise-mode search
            # can find clusters by what users asked, not just assistant answers.
            for cl in clusters:
                member_chunks = [chunk_map[cid] for cid in cl["chunk_ids"] if cid in chunk_map]
                texts: list[str] = []
                for c in member_chunks:
                    if c.assistant_text:
                        texts.append(c.assistant_text)
                    if c.user_text:
                        texts.append(c.user_text)
                    if c.tools_used:
                        texts.append(" ".join(c.tools_used))
                    if c.files_touched:
                        texts.append(" ".join(c.files_touched))
                joined = " ".join(texts)
                if len(joined) > 4000:
                    joined = joined[:4000].rsplit(" ", 1)[0]
                cl["search_text"] = joined

            db.save_clusters(clusters, memberships)
            # Pre-generate concat summaries at build time (read path stays pure).
            # Skip clusters that already have LLM summaries (preserved across rebuilds).
            llm_cluster_ids = {
                r["cluster_id"]
                for r in db._conn.execute(
                    "SELECT cluster_id FROM cluster_summaries WHERE method = 'llm'"
                ).fetchall()
            }
            for cl in clusters:
                if cl["cluster_id"] in llm_cluster_ids:
                    continue  # Already has LLM summary
                member_chunks = [chunk_map[cid] for cid in cl["chunk_ids"] if cid in chunk_map]
                if member_chunks:
                    summary = generate_concat_summary(member_chunks, max_tokens=200)
                    if summary:
                        db.save_cluster_summary(cl["cluster_id"], summary)
            print(f"  Clusters: {len(clusters)} topic clusters from {sum(c['chunk_count'] for c in clusters)} chunks")
        else:
            print("  Clusters: none (chunks may not be related enough)")

    # Auto-tag clusters + build timeline arcs (Phase 10)
    try:
        from synapt.recall.journal import (
            _read_all_entries, _journal_path, _dedup_entries,
        )
        from synapt.recall.tagging import extract_tags as _extract_tags
        from synapt.recall.timeline import (
            build_timeline_clusters,
            save_timeline_clusters,
        )

        # Read journal entries from ALL worktrees (not just local) and
        # dedup by session_id, preferring enriched over auto stubs.
        all_journal_entries: list = []
        for jf in journal_files:
            if jf.exists():
                all_journal_entries.extend(_read_all_entries(jf))
        j_entries = _dedup_entries(all_journal_entries)

        # Tag topic clusters with issue refs, branches, keywords
        if transcript_only:
            tagged = 0
            all_clusters = db.load_clusters()
            for cl in all_clusters:
                if cl["cluster_type"] != "topic":
                    continue
                tags = _extract_tags(cl, j_entries)
                if tags:
                    cl["tags"] = tags
                    new_search = cl.get("search_text", "") + " " + " ".join(tags)
                    db._conn.execute(
                        "UPDATE clusters SET tags = ?, search_text = ? "
                        "WHERE cluster_id = ?",
                        (json.dumps(tags), new_search, cl["cluster_id"]),
                    )
                    tagged += 1
            if tagged:
                db._conn.commit()
                # FTS is kept in sync by the clusters_au trigger on UPDATE;
                # save_timeline_clusters() also does a full FTS rebuild.
                print(f"  Tags: {tagged} topic clusters tagged")

        # Build timeline arcs from session grouping
        timeline = build_timeline_clusters(db, j_entries)
        if timeline:
            save_timeline_clusters(db, timeline)
            print(f"  Timeline: {len(timeline)} arcs")
    except Exception as exc:
        logger.warning("Phase 10 (tagging/timeline) failed: %s", exc, exc_info=True)

    # Process pending promotions (advance tiers based on access stats)
    try:
        from synapt.recall.promotion import process_build_promotions
        promo = process_build_promotions(db)
        promo_total = sum(promo.values())
        if promo_total:
            print(f"  Promotions: {promo['summaries_upgraded']} summaries, "
                  f"{promo['candidates_flagged']} candidates, "
                  f"{promo['knowledge_promoted']} knowledge")
    except Exception:
        pass  # Never fail a build due to promotions

    # THE SUMMARY GRINDER IS NOT RUN FROM A BUILD.
    #
    # `upgrade_large_cluster_summaries` makes LLM calls, so it is unbounded work
    # of a different KIND from everything else here: the rest of a build is
    # local and its cost scales with what changed, while this scales with an
    # external service and runs on every build regardless. It also sat inside a
    # bare `except: pass`, so a build reported success whether it worked, did
    # nothing, or failed -- which is the state that makes a cost invisible.
    #
    # Its explicit home is the `maintain` subcommand, which does not exist yet.
    #
    # WHAT IS AND IS NOT LOST IN THE MEANTIME. Summaries come in TWO TIERS, and
    # every loose sentence about this change has been wrong by collapsing them:
    #
    #   CONCAT is the baseline, and it is unaffected. The clustering step above
    #   pre-generates a concat summary for EVERY cluster on every build, skipping
    #   only those that already hold an LLM one. So no cluster is left without a
    #   summary by this change.
    #
    #   LLM is the upgrade, and it has two triggers. `process_build_promotions`
    #   upgrades by ACCESS TIER and still runs. The removed pass upgraded by
    #   SIZE, independent of access -- which is why its query looked for
    #   clusters where `method = 'llm'` was absent.
    #
    # So the residual is exactly this: a cluster that is large but rarely
    # searched keeps its concat summary and waits for `maintain` to be UPGRADED
    # to LLM quality. It does not go unsummarized.
    #
    # Recorded at this length because two earlier drafts of this comment said
    # "summaries stop being generated" and then "no longer gets a summary" --
    # the same error twice, from writing "summary" unqualified in a system that
    # has two tiers of them.

    # Maintain adaptive memory: decay, archival, log compaction
    try:
        decayed = db.recompute_decay_scores()
        archived = db.archive_cold_clusters()
        compacted = db.compact_access_log()
        parts = []
        if archived:
            parts.append(f"{len(archived)} clusters archived")
        if compacted:
            parts.append(f"{compacted} log entries compacted")
        if parts:
            print(f"  Memory maintenance: {', '.join(parts)}")
    except Exception as exc:
        logger.debug("Memory maintenance failed: %s", exc)

    # Compact + dedup knowledge nodes
    try:
        from synapt.recall.knowledge import (
            dedup_knowledge_nodes, compact_knowledge, _knowledge_path,
        )
        # Always compact first — removes same-ID duplicates from append-only JSONL
        kn_path = _knowledge_path(project_dir)
        if kn_path.exists():
            compacted = compact_knowledge(kn_path)
            if compacted:
                print(f"  Knowledge compact: removed {compacted} stale version(s)")
        # Then merge semantically similar nodes (different IDs, same content)
        merged = dedup_knowledge_nodes(threshold=0.7, project_dir=project_dir)
        if merged:
            print(f"  Knowledge dedup: merged {merged} duplicate(s)")
    except Exception as exc:
        logger.debug("Knowledge dedup failed: %s", exc)

    # Build cross-session links (pre-computed nearest neighbors across sessions)
    if use_embeddings and final_index._all_embeddings:
        try:
            n_links = final_index.build_cross_session_links()
            if n_links:
                print(f"  Cross-session links: {n_links} links across {len(set(c.session_id for c in deduped))} sessions")
        except Exception as exc:
            logger.debug("Cross-session linking failed: %s", exc)

    # Store source file info in DB metadata
    source_files = []
    for build_source in build_sources:
        for fp in sorted(build_source.glob("*.jsonl")):
            st = fp.stat()
            source_files.append({
                "name": fp.name,
                # Exact identity for sidecar projections such as compaction
                # summaries. ``dir`` remains for the transcript parser's
                # backwards-compatible incremental key.
                "source_path": str(fp),
                # The source dir scopes the name. This list is FLAT across every
                # build source, so two worktrees archiving the same session name
                # produce two entries that are indistinguishable without it, and
                # a basename-keyed reader silently keeps only one of them.
                "dir": build_source.name,
                "mtime": st.st_mtime,
                "size": st.st_size,
            })
    # Persist the signature alongside the file list so the NEXT run can tell
    # "nothing changed" from "no idea". Written in the same call, because a
    # manifest that has the file list but not the signature is a state where
    # the no-op check silently never fires.
    # RECOMPUTED, not the pre-build value. The build MUTATES ONE OF ITS OWN
    # SIGNED INPUTS: it synthesizes auto-journal stubs, so the journal on disk
    # after a build differs from the journal it read. Storing the pre-build
    # signature means the next run computes a different digest from an unchanged
    # workspace and the no-op can NEVER fire -- measured: pre 26636950,
    # post f4d4491d, on a workspace nobody touched in between.
    #
    # What is stored is therefore "the input state as of the end of this build",
    # which is exactly what the next run's pre-build signature is compared against.
    #
    # BUT A RECOMPUTE ALSO PICKS UP ARRIVALS. Anything that landed between the
    # index write and this line gets stat'd into the stored signature while
    # never having been read, so the next run compares equal, prints "Up to
    # date", and the content is invisible until something else happens to
    # change the digest. Measured on v3: a transcript dropped after the index
    # save was certified as indexed and returned zero content (Atlas, r2).
    #
    # So the signature is stored ONLY when the read-only inputs are unchanged
    # since the build read them. When they are not, the manifest keeps its file
    # list and carries NO signature, which `signature_from_manifest` resolves to
    # None and `is_noop` resolves to work. Failing toward one unnecessary build
    # is the same asymmetry the no-op check already documents, applied to the
    # one window where the build cannot vouch for its own inputs.
    #
    # TWO WINDOWS, TWO COMPARISONS, AND THE PERSISTED VALUE IS OLDER THAN BOTH.
    #   1. start -> read boundary: read-only inputs must not have moved while
    #      transcripts, channels and the archive were being parsed. Journals are
    #      excluded from THIS one only, because the build writes them in this
    #      window and would otherwise flag its own synthesis as an arrival.
    #   2. read boundary -> now: the full input set, all four classes, must be
    #      unchanged. Recomputed over the CURRENT sets rather than the ones read,
    #      so a file that APPEARED mid-build also fails -- the safe direction.
    # What is stored is `signature_at_read`, sampled before either comparison,
    # so an arrival at any later point -- including after this very check --
    # cannot be inside it. The next run then computes a different digest and
    # does the work.
    #
    # The first comparison is witnessed by the channel-arrival test. It
    # originally guarded mid-parse TRANSCRIPT arrivals; keying the signature on
    # `build_sources` subsumed that case, leaving it responsible for channels
    # and the archive, which are parsed before the journal read boundary.
    # Its witness pins store resolution through the explicit env seam rather
    # than inferring it from cwd -- an earlier attempt inferred, was defeated by
    # its own inference, and was wrongly reported as blocked by test isolation.
    # Compaction handoffs get an explicit continuity-metadata projection, so
    # SessionStart need not rely on their incidental ordinary-turn shape or
    # reopen and scan a transcript.
    compaction_indexed = True
    try:
        from synapt.recall.compaction import update_compaction_summary_index
        update_compaction_summary_index(
            build_sources,
            project=project_dir,
            previous_manifest=incremental_manifest,
        )
    except Exception as exc:
        compaction_indexed = False
        logger.warning("Compaction summary indexing failed: %s", exc)

    manifest_payload = {"source_files": source_files}
    inputs_stable = (
        readonly_at_read.digest == readonly_before.digest
        and compute_input_signature(
            source_dirs=build_sources,
            channels_dir=_channels_dir(project_dir),
            journal_paths=_build_journal_files(project_dir),
            archive_paths=archive_paths,
        ).digest == signature_at_read.digest
    )
    if inputs_stable and compaction_indexed:
        manifest_payload.update(signature_to_manifest(signature_at_read))
    elif not compaction_indexed:
        print("  Note: compaction summary indexing failed; next run will not skip")
    else:
        print("  Note: input changed during the build; next run will not skip")
    db.save_manifest(manifest_payload)

    if progress:
        progress("finalizing")
    logger.info("build: complete in %.1fs (%d chunks)", _time.monotonic() - build_t0, len(deduped))
    return final_index


def discover_transcript_dirs() -> list[Path]:
    """Find all Claude Code project transcript directories.

    Scans ~/.claude/projects/*/ for directories containing .jsonl files.
    Returns directories sorted alphabetically.
    """
    base = Path.home() / ".claude" / "projects"
    if not base.exists():
        return []
    dirs = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and any(d.glob("*.jsonl")):
            dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_SPLIT_EXPERIMENTAL_WARNING = (
    "WARNING: sharded split mode is still experimental. "
    "Verify end-to-end sharded search results before treating split shards as production-safe."
)


def cmd_split(args: argparse.Namespace) -> None:
    """Split monolithic recall.db into quarterly shards."""
    from synapt.recall.sharding import split_monolithic_db, estimate_split
    index_dir = project_index_dir()
    print(_SPLIT_EXPERIMENTAL_WARNING, file=sys.stderr)

    if args.dry_run:
        plan = split_monolithic_db(index_dir, dry_run=True)
        print("Split plan (dry run):")
        total = 0
        for name, count in sorted(plan.items()):
            if name == "index.db":
                print(f"  {name}: knowledge, clusters, metadata")
            else:
                print(f"  {name}: {count} chunks")
                total += count
        print(f"  Total: {total} chunks across {len(plan) - 1} shard(s)")
        return

    try:
        plan = split_monolithic_db(index_dir)
        total = sum(v for k, v in plan.items() if k != "index.db")
        shards = len(plan) - 1  # Exclude index.db
        print(f"Split complete: {total} chunks across {shards} quarterly shard(s)")
        for name, count in sorted(plan.items()):
            if name != "index.db":
                print(f"  {name}: {count} chunks")
        print(f"\nOriginal recall.db preserved. Delete it manually after verifying.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_build(args: argparse.Namespace) -> None:
    """Build a transcript index from local files, HuggingFace, or ChatGPT."""
    project = Path.cwd().resolve()
    use_emb = not args.no_embeddings

    # Re-scrub archives if requested
    if getattr(args, "rescrub", False):
        from synapt.recall.scrub import scrub_jsonl as _rescrub_jsonl

        archive_dirs = list(all_worktree_archive_dirs(project))
        archive_dir = project_archive_dir(project)
        if archive_dir.exists() and archive_dir.resolve() not in {p.resolve() for p in archive_dirs}:
            archive_dirs.append(archive_dir)

        total = 0
        for ad in archive_dirs:
            for jsonl_file in sorted(ad.glob("*.jsonl")):
                _rescrub_jsonl(jsonl_file)
                total += 1
        if total:
            print(f"[build] Re-scrubbed {total} archived transcript(s)")
            args.incremental = False  # Force full rebuild after rescrub

    # Check for legacy index
    legacy = _check_legacy_index()
    if legacy:
        print(f"[build] Note: legacy index found at {legacy}")
        print(f"[build] New location: {project_index_dir()}")
        print()

    from synapt.recall.codex import _has_buildable_transcripts

    source_dirs: list[Path] = []
    if args.source:
        for src in args.source:
            source_dirs.append(Path(src).expanduser())

    if args.hf:
        hf_dir = _download_hf_transcripts(args.hf)
        if hf_dir:
            source_dirs.append(hf_dir)

    if not source_dirs and not args.chatgpt_archive:
        auto_dirs = project_transcript_dirs()
        if auto_dirs:
            source_dirs = auto_dirs
            for td in auto_dirs:
                print(f"[build] Found project transcripts at {td}")
        elif not all_worktree_archive_dirs(project) and not _has_buildable_transcripts(project):
            # No live Claude transcripts, no archived transcripts, AND no
            # discoverable Codex sessions for this project — nothing to build.
            #
            # The Codex arm is the one this guard was missing. `_archive_and_build`
            # runs `archive_codex_transcripts` unconditionally, so a codex-only
            # project HAD work to do and this pre-check refused it before the
            # sweep ever ran. A guard must ask the same question as the step it
            # gates; this one asked a narrower one and won.
            print("Error: no transcripts found for current project.", file=sys.stderr)
            print("Specify --source, --hf, or --chatgpt-archive explicitly.", file=sys.stderr)
            sys.exit(1)

    if use_emb:
        print("[build] Computing embeddings (this may take a moment) ...")

    final_index = _archive_and_build(
        project,
        source_dirs=source_dirs or None,
        use_embeddings=use_emb,
        incremental=args.incremental,
        chatgpt_archive=args.chatgpt_archive,
    )

    if not final_index:
        print("Error: no chunks found.", file=sys.stderr)
        sys.exit(1)

    stats = final_index.stats()
    print(f"\n[build] Done!")
    print(f"  Chunks: {stats['chunk_count']}")
    print(f"  Sessions: {stats['session_count']}")
    if stats.get("date_range"):
        print(f"  Date range: {stats['date_range']['earliest'][:10]} -> {stats['date_range']['latest'][:10]}")
    print(f"  Saved to: {project_index_dir()}")


def cmd_search(args: argparse.Namespace) -> None:
    """Search the transcript index."""
    import time
    from synapt.recall.config import load_config

    profile = getattr(args, "profile", False)

    index_dir = _resolve_index_dir(args)
    if not (index_dir / "recall.db").exists() and not (index_dir / "chunks.jsonl").exists():
        print(f"Error: no index found at {index_dir}", file=sys.stderr)
        print("Run 'synapt build' or 'synapt setup' first.", file=sys.stderr)
        sys.exit(1)

    # Resolve max_tokens: CLI flag → config → default (500 for CLI)
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = min(500, load_config().get_max_tokens())

    t0 = time.perf_counter()
    index = TranscriptIndex.load(index_dir, use_embeddings=True)
    t_load = time.perf_counter()

    result = index.lookup(
        args.query,
        max_chunks=args.max_chunks,
        max_tokens=max_tokens,
        max_sessions=args.max_sessions,
        after=args.after,
        before=args.before,
    )
    t_search = time.perf_counter()

    if result:
        print(result)
    else:
        print("No results found.")

    if profile:
        t_total = t_search - t0
        emb_status = index._embedding_status
        emb_loaded = getattr(index, "_embeddings_loaded", "n/a")
        chunk_count = len(index.chunks)
        emb_matrix = getattr(index, "_emb_matrix", None)
        numpy_info = f"{emb_matrix.shape[0]}x{emb_matrix.shape[1]} float32" if emb_matrix is not None else "dict"
        print(f"\n--- Profile ---", file=sys.stderr)
        print(f"  Index load:    {t_load - t0:.3f}s", file=sys.stderr)
        print(f"  Search:        {t_search - t_load:.3f}s", file=sys.stderr)
        print(f"  Total:         {t_total:.3f}s", file=sys.stderr)
        print(f"  Chunks:        {chunk_count}", file=sys.stderr)
        print(f"  Embeddings:    {emb_status} (loaded: {emb_loaded}, storage: {numpy_info})", file=sys.stderr)
        print(f"  FTS backend:   {'sqlite' if index._rowid_to_idx else 'bm25'}", file=sys.stderr)


_DEFAULT_BENCHMARK_QUERIES = [
    "what was discussed last session",
    "how does authentication work",
    "debug error in production",
    "configuration settings",
    "performance optimization",
    "test coverage for module",
    "API endpoint changes",
    "migration plan",
]


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run search pipeline benchmarks and report timing statistics."""
    import time

    index_dir = Path(args.index) if args.index else _resolve_index_dir(args)
    if not (index_dir / "recall.db").exists() and not (index_dir / "chunks.jsonl").exists():
        print(f"Error: no index found at {index_dir}", file=sys.stderr)
        sys.exit(1)

    iterations = args.iterations
    queries = args.queries.split(";") if args.queries else _DEFAULT_BENCHMARK_QUERIES
    json_output = args.json_output

    # Phase 1: Cold start (index load)
    t0 = time.perf_counter()
    index = TranscriptIndex.load(index_dir, use_embeddings=True)
    t_cold = (time.perf_counter() - t0) * 1000  # ms

    chunk_count = len(index.chunks)
    session_count = len(index.sessions)
    emb_status = index._embedding_status
    emb_matrix = getattr(index, "_emb_matrix", None)
    numpy_storage = emb_matrix is not None

    # Memory footprint estimate (rough: list container + ~1KB per chunk for strings)
    import sys as _sys
    mem_bytes = _sys.getsizeof(index.chunks)
    mem_bytes += chunk_count * 1024  # ~1KB average per chunk (strings, metadata)
    if emb_matrix is not None:
        mem_bytes += emb_matrix.nbytes
    all_emb = getattr(index, "_all_embeddings", {})
    if all_emb:
        mem_bytes += _sys.getsizeof(all_emb)
    mem_mb = mem_bytes / (1024 * 1024)

    # Phase 2: Query latencies
    query_results = []
    for q in queries:
        latencies = []
        for _ in range(iterations):
            # Clear query cache between iterations
            index._query_cache.clear()
            t_start = time.perf_counter()
            result = index.lookup(q, max_chunks=5, max_tokens=500)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)  # ms

        latencies.sort()
        n = len(latencies)
        p50 = latencies[n // 2]
        p95 = latencies[int(n * 0.95)] if n >= 20 else latencies[-1]
        p99 = latencies[int(n * 0.99)] if n >= 100 else latencies[-1]
        mean = sum(latencies) / n
        has_results = bool(result)

        query_results.append({
            "query": q,
            "mean_ms": round(mean, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "min_ms": round(latencies[0], 2),
            "max_ms": round(latencies[-1], 2),
            "has_results": has_results,
        })

    # Phase 3: Second load (warm filesystem cache)
    index2 = None
    t_w0 = time.perf_counter()
    index2 = TranscriptIndex.load(index_dir, use_embeddings=True)
    t_warm = (time.perf_counter() - t_w0) * 1000
    del index2

    # Output
    if json_output:
        output = {
            "cold_start_ms": round(t_cold, 2),
            "warm_start_ms": round(t_warm, 2),
            "queries": query_results,
            "index_info": {
                "chunk_count": chunk_count,
                "session_count": session_count,
                "embedding_status": emb_status,
                "numpy_storage": numpy_storage,
                "memory_mb": round(mem_mb, 2),
                "index_dir": str(index_dir),
            },
            "config": {
                "iterations": iterations,
                "query_count": len(queries),
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print("Recall Search Benchmark")
        print("=" * 50)
        print(f"  Index:         {index_dir}")
        print(f"  Chunks:        {chunk_count}")
        print(f"  Sessions:      {session_count}")
        print(f"  Embeddings:    {emb_status} ({'numpy' if numpy_storage else 'dict'})")
        print(f"  Memory:        {mem_mb:.1f} MB")
        print()
        print(f"  Cold start:    {t_cold:.0f} ms")
        print(f"  Warm start:    {t_warm:.0f} ms")
        print()
        print(f"  Query Latencies ({iterations} iterations)")
        print(f"  {'Query':<35} {'p50':>7} {'p95':>7} {'mean':>7}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7}")
        for qr in query_results:
            label = qr["query"][:33] + ".." if len(qr["query"]) > 35 else qr["query"]
            hit = "" if qr["has_results"] else " (no hits)"
            print(f"  {label:<35} {qr['p50_ms']:>6.1f}ms {qr['p95_ms']:>6.1f}ms {qr['mean_ms']:>6.1f}ms{hit}")
        print()

        # Summary
        all_means = [qr["mean_ms"] for qr in query_results]
        overall_p50 = sorted(all_means)[len(all_means) // 2]
        print(f"  Overall p50:   {overall_p50:.1f} ms")
        print(f"  Total queries: {len(queries) * iterations}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show index statistics."""
    index_dir = _resolve_index_dir(args)
    if not (index_dir / "recall.db").exists() and not (index_dir / "manifest.json").exists():
        print(f"Error: no index found at {index_dir}", file=sys.stderr)
        sys.exit(1)

    index = TranscriptIndex.load(index_dir, use_embeddings=False)
    stats = index.stats()

    # Load manifest from DB or legacy file
    manifest: dict = {}
    try:
        if index._db:
            manifest = index._db.load_manifest()
        elif (index_dir / "manifest.json").exists():
            with open(index_dir / "manifest.json", encoding="utf-8") as f:
                manifest = json.load(f)
    except Exception:
        pass

    print("Transcript Index Stats")
    print("=" * 40)
    print(f"  Project:          {Path.cwd()}")
    print(f"  Index:            {index_dir}")
    print(f"  Chunks:           {stats.get('chunk_count', 0)}")
    print(f"  Sessions:         {stats.get('session_count', 0)}")
    print(f"  Avg chunks/sess:  {stats.get('avg_chunks_per_session', 0):.1f}")
    if stats.get("date_range"):
        dr = stats["date_range"]
        print(f"  Date range:       {dr['earliest'][:10]} -> {dr['latest'][:10]}")
    print(f"  Unique tools:     {stats.get('total_tools_used', 0)}")
    print(f"  Unique files:     {stats.get('total_files_touched', 0)}")
    print(f"  Built:            {manifest.get('build_timestamp', 'unknown')[:19]}")

    total_size = sum(fp.stat().st_size for fp in index_dir.iterdir() if fp.is_file())
    print(f"  Index size:       {format_size(total_size)}")

    # Show archive stats
    archive_dir = project_archive_dir()
    if archive_dir.exists():
        archive_files = list(archive_dir.glob("*.jsonl"))
        archive_size = sum(f.stat().st_size for f in archive_files)
        print(f"  Archived:         {len(archive_files)} transcripts ({format_size(archive_size)})")

    source_files = manifest.get("source_files", [])
    if source_files:
        total_source = sum(sf.get("size", 0) for sf in source_files)
        print(f"  Source files:     {len(source_files)} ({format_size(total_source)})")

    # Cluster stats
    if index._db:
        n_clusters = index._db.cluster_count()
        if n_clusters > 0:
            print(f"  Clusters:         {n_clusters}")

    # Active model configuration
    try:
        from synapt.recall.config import load_config
        cfg = load_config()
        models = cfg.active_models()
        print()
        print("Active Models")
        print("-" * 40)
        for key, model in models.items():
            print(f"  {key:16s}  {model}")
        if cfg.backend != "auto":
            print(f"  {'backend':16s}  {cfg.backend}")
    except Exception as e:
        logger.debug("Failed to load model config: %s", e)


def cmd_sessions(args: argparse.Namespace) -> None:
    """List recent sessions with date, turn count, and first message."""
    from synapt.recall.sharding import is_sharded

    index_dir = _resolve_index_dir(args)
    if (
        not (index_dir / "recall.db").exists()
        and not (index_dir / "chunks.jsonl").exists()
        and not is_sharded(index_dir)
    ):
        print(f"Error: no index found at {index_dir}", file=sys.stderr)
        print("Run 'synapt build' or 'synapt setup' first.", file=sys.stderr)
        sys.exit(1)

    from synapt.recall.resume import load_resume_index

    index = load_resume_index(index_dir)
    try:
        sessions = index.list_sessions(
            max_sessions=args.max_sessions,
            after=args.after,
            before=args.before,
        )
    finally:
        db = getattr(index, "_db", None)
        if db is not None:
            db.close()

    if not sessions:
        print("No sessions found.")
        return

    print(f"Recent sessions ({len(sessions)}):")
    for s in sessions:
        print(
            f"  {s['date']}  {s['session_id'][:8]}  "
            f"{s['turn_count']} turns  {s['files_count']} files  "
            f"[{s['source_root']}]  \"{s['first_message']}\""
        )


def cmd_resume(args: argparse.Namespace) -> None:
    """Print the tail of a session so a fresh session can pick up where it stopped.

    Three outcomes are kept distinguishable because they have different fixes:
    no index at all (exit 1 — build one), an index with no sessions (exit 0 —
    nothing to resume), and a session id that does not resolve (exit 1 — the
    request was wrong). Collapsing them would send the reader down the wrong path.
    """
    from synapt.recall.journal import _journal_path
    from synapt.recall.resume import (
        ResumeError,
        build_resume_view,
        caller_transcripts,
        format_resume,
        load_resume_index,
    )
    from synapt.recall.sharding import is_sharded

    index_dir = _resolve_index_dir(args)
    if (
        not (index_dir / "recall.db").exists()
        and not (index_dir / "chunks.jsonl").exists()
        and not is_sharded(index_dir)
    ):
        print(f"Error: no index found at {index_dir}", file=sys.stderr)
        print("Run 'synapt recall build' or 'synapt init' first.", file=sys.stderr)
        sys.exit(1)

    index = load_resume_index(index_dir)

    try:
        try:
            view = build_resume_view(
                index,
                session_id=getattr(args, "session", None),
                limit=getattr(args, "turns", None) or 10,
                journal_path=_journal_path(),
                caller_sources=caller_transcripts(
                    getattr(args, "project", None) or Path.cwd()
                ),
                agent_id=os.environ.get("SYNAPT_AGENT_ID"),
            )
        except ResumeError as exc:
            # An empty index is an honest empty state, not a failure to act on.
            if not index._session_order:
                print("No sessions indexed yet. Nothing to resume.")
                return
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    finally:
        db = getattr(index, "_db", None)
        if db is not None:
            db.close()

    # Freshness is attached AFTER the view is built, so build_resume_view keeps
    # its no-implicit-I/O contract and the check can never change what is shown
    # -- only what the reader is told about it.
    #
    # The deep leg runs on the SUSPICIOUS COMBINATION: the cheap leg says fresh
    # and the view is empty. That is exactly when the deep leg's cost (~1.2 s on
    # the store it was measured against) is worth spending,
    # because it is the moment an "empty" verdict either becomes load-bearing
    # or turns out to be an un-archived session. Everywhere else the cheap leg
    # (~24 ms there) answers, and a stale verdict needs no second opinion.
    view = _attach_freshness(view, args)
    view = _attach_unclean_end(view, args)

    print(format_resume(view))


def _attach_unclean_end(view, args):
    """Judge the SELECTED session, not the newest transcript: the process
    running `synapt resume` is usually itself the newest transcript.
    Never raises; None means NOT CHECKED."""
    try:
        from synapt.recall.journal import _journal_path
        from synapt.recall.resume import (
            authored_journals,
            caller_transcripts,
            detect_unclean_end,
        )
        from synapt.checkpoint import read_checkpoint

        project = getattr(args, "project", None) or Path.cwd()
        sources = [
            item for item in caller_transcripts(project)
            if item.session_id == view.session_id
        ]
        view.unclean_end = detect_unclean_end(
            sources,
            checkpoint=read_checkpoint(Path(project)),
            # Ambient journal store when the caller did not pass a deliberate
            # project (the wake case): resolve the session-consistent store via
            # the env override rather than cwd. An explicit args.project stays the
            # deliberate export/import target. (dual-use wake fix)
            authored_journals=authored_journals(
                _journal_path(getattr(args, "project", None))
            ),
        )
    except Exception:
        pass
    return view


def _attach_freshness(view, args):
    """Return *view* with an index-freshness verdict attached.

    Never raises: a failure to compute freshness must not break the command it
    annotates. On failure the verdict stays ``None``, which the renderer treats
    as NOT CHECKED rather than as fresh.
    """
    import dataclasses

    from synapt.recall.freshness import check_index_freshness

    # Bind to the index the RENDER loaded, not to a separately-resolved
    # project. `resume` has no --project, so resolving one here meant freshness
    # answered about the cwd's store while the view came from --index: a real
    # stale index could be reported as fine.
    index_dir = _resolve_index_dir(args)
    project = getattr(args, "project", None)
    try:
        result = check_index_freshness(project, index_dir=index_dir)
        if not result.stale and not view.turns:
            result = check_index_freshness(project, index_dir=index_dir, deep=True)
    except Exception:
        return view
    return dataclasses.replace(view, freshness=result)


def cmd_rebuild(args: argparse.Namespace) -> None:
    """Incremental rebuild triggered by hooks. Auto-discovers current project."""
    project = Path.cwd().resolve()

    if not project_transcript_dirs(project):
        return

    final_index = _archive_and_build(
        project,
        use_embeddings=False,
        incremental=True,
    )

    if final_index:
        stats = final_index.stats()
        print(f"synapt: rebuilt index ({stats['chunk_count']} chunks)", file=sys.stderr)

    # Optional sync after rebuild
    if getattr(args, "sync", False):
        _sync_after_rebuild(project)

    # Optional enrichment of auto-stubs
    enrich_n = getattr(args, "enrich", 0)
    if enrich_n and final_index:
        try:
            from synapt.recall.enrich import enrich_all, _MLX_AVAILABLE
            if _MLX_AVAILABLE:
                count = enrich_all(
                    project_dir=project,
                    max_entries=enrich_n,
                )
                if count:
                    print(f"  Enriched {count} journal stub(s)", file=sys.stderr)
        except Exception:
            pass  # Enrichment is best-effort; don't break the hook


def cmd_rescrub(args: argparse.Namespace) -> None:
    """Re-scrub archived transcripts with updated secret patterns.

    Runs scrub_jsonl() on all archived transcript files in-place, then
    does a full (non-incremental) rebuild so the index reflects the
    cleaned transcripts. Use after updating scrub patterns to retroactively
    remove secrets that slipped through earlier builds.
    """
    from synapt.recall.scrub import scrub_jsonl

    project = Path.cwd().resolve()
    archive_dirs = list(all_worktree_archive_dirs(project))
    archive_dir = project_archive_dir(project)
    if archive_dir.exists() and archive_dir.resolve() not in {p.resolve() for p in archive_dirs}:
        archive_dirs.append(archive_dir)

    if not archive_dirs:
        print("No archived transcripts found.", file=sys.stderr)
        sys.exit(1)

    total = 0
    for ad in archive_dirs:
        for jsonl_file in sorted(ad.glob("*.jsonl")):
            scrub_jsonl(jsonl_file)  # in-place
            total += 1
    print(f"[rescrub] Scrubbed {total} archived transcript(s)")

    if not args.no_rebuild:
        print("[rescrub] Rebuilding index from scrubbed transcripts ...")
        use_emb = not getattr(args, "no_embeddings", False)
        final_index = _archive_and_build(
            project,
            use_embeddings=use_emb,
            incremental=False,  # Full rebuild — must re-index everything
        )
        if final_index:
            stats = final_index.stats()
            print(f"[rescrub] Done! {stats['chunk_count']} chunks, {stats['session_count']} sessions")
        else:
            print("[rescrub] Warning: rebuild produced no chunks", file=sys.stderr)
    else:
        print("[rescrub] Skipping rebuild (--no-rebuild). Run 'synapt recall build' manually.")


def _sync_after_rebuild(project: Path) -> None:
    """Push new transcripts to HF if sync is configured and debounce allows."""
    from synapt.recall.archive import load_sync_config, upload_to_hf, should_sync

    config = load_sync_config(project)
    sync = config.get("sync", {})
    if sync.get("provider") == "hf" and sync.get("auto_sync") and sync.get("repo_id"):
        if not should_sync(project):
            return
        extra = sync.get("extra_files", [])
        uploaded = upload_to_hf(project, sync["repo_id"], extra_files=extra)
        if uploaded:
            print(f"synapt: synced {uploaded} file(s) to HF", file=sys.stderr)


def cmd_archive(args: argparse.Namespace) -> None:
    """Archive transcripts locally without indexing."""
    from synapt.recall.archive import archive_transcripts

    project = Path.cwd().resolve()
    transcript_all = project_transcript_dirs(project)
    if not transcript_all:
        return  # New project, no transcripts yet — not an error

    total_copied: list[str] = []
    for transcript_dir in transcript_all:
        copied = archive_transcripts(project, transcript_dir)
        if copied:
            total_copied.extend(copied)
            print(f"Archived {len(copied)} transcript(s) from {transcript_dir}", file=sys.stderr)
    if not total_copied:
        print("All transcripts already archived.", file=sys.stderr)


def cmd_export(args: argparse.Namespace) -> None:
    """Export portable recall state to a .synapt-archive file."""
    from synapt.recall.archive import export_recall_archive

    # Do NOT default to Path.cwd() here: an explicit root suppresses the
    # SYNAPT_RECALL_ROOT / GRIPSPACE_ROOT overrides inside project_data_dir.
    # None means "resolve like every other recall verb".
    project = Path(args.path).expanduser().resolve() if getattr(args, "path", None) else None
    try:
        output_path, manifest = export_recall_archive(
            project,
            Path(args.output).expanduser() if args.output else None,
            exclude_transcripts=args.exclude_transcripts,
            exclude_channels=args.exclude_channels,
        )
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Name the resolved store (recall#963 reporting contract): a bare count is
    # exactly the report shape that let a wrong-store export hide.
    print(f"Exported recall archive to {output_path}")
    print(f"  store={manifest.get('data_dir', '?')}")
    print(
        f"  chunks={manifest.get('chunk_count', 0)} "
        f"knowledge={manifest.get('knowledge_count', 0)} "
        f"worktrees={manifest.get('worktree_count', 0)}"
    )


def cmd_import(args: argparse.Namespace) -> None:
    """Import portable recall state from a .synapt-archive file."""
    from synapt.recall.archive import import_recall_archive

    # Same rule as export: None resolves via SYNAPT_RECALL_ROOT / GRIPSPACE_ROOT
    # + inference.
    project = Path(args.path).expanduser().resolve() if getattr(args, "path", None) else None
    mode = "merge" if args.merge else "replace"
    try:
        summary = import_recall_archive(
            project,
            Path(args.archive),
            mode=mode,
        )
    except Exception as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Imported recall archive from {Path(args.archive).expanduser().resolve()}")
    print(
        f"  mode={summary.get('mode', mode)} "
        f"chunks={summary.get('chunk_count', 0)} "
        f"knowledge={summary.get('knowledge_count', 0)} "
        f"store={summary.get('data_dir', '?')}"
    )


def cmd_transcript(args: argparse.Namespace) -> None:
    """Display or save a session transcript."""
    from synapt.recall.archive import archive_transcripts

    project = Path.cwd().resolve()
    transcript_dir = project_transcript_dir(project)

    if not transcript_dir:
        print("No transcript directory found for this project.", file=sys.stderr)
        sys.exit(1)

    # Find the transcript file
    if args.session_id:
        target = transcript_dir / f"{args.session_id}.jsonl"
        if not target.exists():
            # Also check archive
            archive = project_archive_dir(project)
            target = archive / f"{args.session_id}.jsonl"
        if not target.exists():
            print(f"Session not found: {args.session_id}", file=sys.stderr)
            # List available sessions
            _list_available_sessions(transcript_dir, project)
            sys.exit(1)
    else:
        # Find current/most recent session by mtime
        jsonl_files = sorted(transcript_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not jsonl_files:
            print("No transcript files found.", file=sys.stderr)
            sys.exit(1)
        target = jsonl_files[0]

    # --save: archive it locally
    if args.save:
        copied = archive_transcripts(project, transcript_dir)
        if copied:
            print(f"Archived {len(copied)} transcript(s) to .synapt/recall/transcripts/", file=sys.stderr)
        else:
            print("Already archived.", file=sys.stderr)
        return

    # --list: show available sessions
    if args.list:
        _list_available_sessions(transcript_dir, project)
        return

    # Display the transcript
    session_id = target.stem
    size = target.stat().st_size
    print(f"Session: {session_id}")
    print(f"File: {target}")
    print(f"Size: {format_size(size)}")
    print("=" * 60)

    turn_count = 0
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if _is_real_user_message(entry):
                user_text = _extract_user_text(entry)
                if not user_text or "<command-name>" in user_text:
                    continue
                turn_count += 1
                ts = entry.get("timestamp", "")
                header = f"--- Turn {turn_count} ---"
                if ts:
                    header = f"--- Turn {turn_count} [{ts[:19]}] ---"
                print(f"\n{header}")
                print(f"User: {user_text[:2000]}")

            elif entry.get("type") == "assistant":
                text, tools, _files = _extract_assistant_content(entry)
                if text or tools:
                    if text:
                        print(f"Assistant: {text[:2000]}")
                    if tools:
                        print(f"  [Tools: {', '.join(tools)}]")

    print(f"\n{'=' * 60}")
    print(f"Total turns: {turn_count}")


def _list_available_sessions(transcript_dir: Path, project: Path) -> None:
    """List available transcript sessions."""
    archive = project_archive_dir(project)

    # Gather all unique session IDs from both source and archive
    sessions: dict[str, tuple[float, int, str]] = {}  # id -> (mtime, size, location)

    for d, label in [(transcript_dir, "live"), (archive, "archived")]:
        if not d.exists():
            continue
        for f in d.glob("*.jsonl"):
            sid = f.stem
            st = f.stat()
            if sid not in sessions or st.st_mtime > sessions[sid][0]:
                sessions[sid] = (st.st_mtime, st.st_size, label)

    if not sessions:
        print("No sessions found.")
        return

    print(f"{'Session ID':<40} {'Size':>10} {'Location':>10}  Modified")
    print("-" * 90)
    for sid, (mtime, size, loc) in sorted(sessions.items(), key=lambda x: x[1][0], reverse=True):
        dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{sid:<40} {format_size(size):>10} {loc:>10}  {dt}")


def cmd_journal(args: argparse.Namespace) -> None:
    """Display or write session journal entries."""
    from synapt.recall.journal import (
        append_entry,
        auto_extract_entry,
        format_entry_full,
        format_for_session_start,
        format_write_confirmation,
        latest_transcript_path,
        merge_carried_forward_with_report,
        read_entries,
        read_latest,
        read_previous_meaningful,
    )

    if args.read:
        entry = read_latest(meaningful=True)
        if not entry:
            return  # Silent — no meaningful journal yet (hook context)
        text = format_for_session_start(entry)
        if text:
            print(text)
        return

    if args.list:
        n = args.show if args.show else 5
        entries = read_entries(n=n)
        if not entries:
            print("No journal entries yet.")
            return
        for i, entry in enumerate(entries):
            if i > 0:
                print("\n---\n")
            print(format_entry_full(entry))
        return

    if args.show:
        if args.show < 1:
            print("--show requires a positive integer.", file=sys.stderr)
            sys.exit(1)
        entries = read_entries(n=args.show)
        if not entries:
            print("No journal entries yet.")
            return
        if args.show > len(entries):
            print(f"Only {len(entries)} journal entries exist.", file=sys.stderr)
        idx = min(args.show - 1, len(entries) - 1)
        print(format_entry_full(entries[idx]))
        return

    if getattr(args, "repair", False):
        from synapt.recall.journal import (
            _journal_path, format_repair_report, repair_journal, sweep_stores,
        )
        from synapt.recall.core import project_data_dir

        dry = getattr(args, "dry_run", False)
        explicit = getattr(args, "path", None)

        # Every line below names the store it examined. A bare "nothing to
        # repair" is indistinguishable from having examined the wrong store,
        # and the data root is resolved from the working directory — so a desk
        # whose writes land under a different root gets a clean, false report.
        if getattr(args, "all_stores", False):
            root = Path(explicit) if explicit else project_data_dir()
            reports = sweep_stores(root, dry_run=dry)
        elif explicit:
            reports = [repair_journal(Path(explicit), dry_run=dry)]
        else:
            reports = [repair_journal(_journal_path(), dry_run=dry)]

        for report in reports:
            print(format_repair_report(report))

        repaired = sum(r["repaired_entries"] for r in reports)
        if dry and repaired:
            print(f"\nDry run — nothing written. {repaired} value(s) would be "
                  f"recovered. Re-run without --dry-run to apply.")
        return

    if not args.write:
        print(
            "Usage: synapt recall journal "
            "[--read | --write | --list | --show N | --repair]",
            file=sys.stderr,
        )
        print("  --write is required to create a journal entry.", file=sys.stderr)
        sys.exit(1)

    # --write: auto-extract + merge CLI args
    project = Path.cwd().resolve()
    transcript_path = latest_transcript_path(project)
    entry = auto_extract_entry(transcript_path=transcript_path, cwd=str(project))
    previous_entry = read_previous_meaningful(entry.session_id)

    # Merge CLI-provided fields
    if args.focus:
        entry.focus = args.focus
    if args.done:
        entry.done = split_journal_field(args.done)
    if args.decisions:
        entry.decisions = split_journal_field(args.decisions)
    explicit_next_steps = list(entry.next_steps)
    if args.next:
        entry.next_steps = split_journal_field(args.next)
        explicit_next_steps = list(entry.next_steps)
    entry.next_steps, carry_report = merge_carried_forward_with_report(
        entry.next_steps,
        entry.done,
        previous_entry,
    )

    # Clear auto flag if user provided rich content
    if entry.has_rich_content():
        entry.auto = False

    # Persist auto-stubs even without rich content — enrich fills them in later.
    # Session-start display already filters to has_rich_content() entries.
    if entry.auto and not entry.has_rich_content():
        if not entry.has_content():
            print("No content to journal (no files modified, no fields provided).", file=sys.stderr)
            return
        path = append_entry(entry)
        sid = entry.session_id[:8] if entry.session_id else "unknown"
        print(f"Auto-stub saved for enrichment ({sid})", file=sys.stderr)
        return
    # Skip completely empty entries (no files, no fields)
    if not entry.has_content():
        print("No content to journal (no files modified, no fields provided).", file=sys.stderr)
        return

    path = append_entry(entry)
    print(f"Journal entry written to {path}", file=sys.stderr)
    print(format_write_confirmation(entry, explicit_next_steps, report=carry_report))


def cmd_enrich(args: argparse.Namespace) -> None:
    """Enrich auto-generated journal stubs using a local MLX model."""
    project = Path.cwd().resolve()
    model = args.model

    init_from = getattr(args, "init_from", None)
    if init_from:
        from synapt.recall.enrich import enrich_transcript_segments
        transcript_path = Path(init_from).expanduser().resolve()
        if not transcript_path.exists():
            print(f"[enrich] Transcript not found: {transcript_path}", file=sys.stderr)
            return
        gap = getattr(args, "gap_minutes", 60)
        print(f"[init] Segmenting transcript: {transcript_path.name}")
        print(f"[init] Gap threshold: {gap} minutes")
        count = enrich_transcript_segments(
            transcript_path=transcript_path,
            project_dir=project,
            model=model,
            dry_run=args.dry_run,
            max_entries=args.max_entries,
            adapter_path=getattr(args, "adapter_path", ""),
            gap_minutes=gap,
        )
        if count:
            action = "would be enriched" if args.dry_run else "enriched"
            print(f"\n[init] Done! {count} segments {action}.")
        else:
            print("[init] No segments to enrich.")
        return

    from synapt.recall.enrich import enrich_all
    print(f"[enrich] Enriching auto-journal stubs with {model} ...")

    count = enrich_all(
        project_dir=project,
        model=model,
        dry_run=args.dry_run,
        max_entries=args.max_entries,
        adapter_path=getattr(args, "adapter_path", ""),
    )

    if count:
        action = "would be enriched" if args.dry_run else "enriched"
        print(f"\n[enrich] Done! {count} entries {action}.")
    else:
        print("[enrich] No entries to enrich (all sessions already have journal entries).")


def cmd_consolidate(args: argparse.Namespace) -> None:
    """Extract durable knowledge from journal entries (memory consolidation)."""
    if getattr(args, "show", False):
        from synapt.recall.knowledge import read_nodes, format_knowledge_for_display
        nodes = read_nodes()
        if not nodes:
            print("No knowledge nodes yet. Run `synapt recall consolidate` to extract knowledge from journal entries.")
            return
        active = [n for n in nodes if n.status == "active"]
        other = [n for n in nodes if n.status != "active"]
        print(f"Knowledge nodes: {len(active)} active, {len(other)} inactive\n")
        if active:
            print("Active:")
            print(format_knowledge_for_display(active))
        if other:
            print("\nInactive:")
            print(format_knowledge_for_display(other))
        return

    from synapt.recall.consolidate import consolidate

    project = Path.cwd().resolve()
    model = args.model

    print(f"[consolidate] Analyzing journal entries with {model} ...")

    result = consolidate(
        project_dir=project,
        model=model,
        dry_run=args.dry_run,
        force=args.force,
        min_entries=args.min_entries,
        adapter_path=getattr(args, "adapter_path", ""),
    )

    if args.dry_run:
        if result.clusters_found:
            print(f"\n[consolidate] Dry run: {result.entries_processed} entries, "
                  f"{result.clusters_found} clusters found.")
        else:
            print(f"\n[consolidate] Dry run: {result.entries_processed} entries, "
                  f"no clusters found (entries may not be related enough).")
        return

    parts = []
    if result.nodes_created:
        parts.append(f"{result.nodes_created} created")
    if result.nodes_corroborated:
        parts.append(f"{result.nodes_corroborated} corroborated")
    if result.nodes_contradicted:
        parts.append(f"{result.nodes_contradicted} contradicted")
    if result.nodes_contested:
        parts.append(f"{result.nodes_contested} contested")
    if result.nodes_deduped:
        parts.append(f"{result.nodes_deduped} deduped")

    if parts:
        print(f"\n[consolidate] Done! Knowledge nodes: {', '.join(parts)}.")
    else:
        print(f"\n[consolidate] No knowledge extracted from {result.entries_processed} entries.")


def cmd_remind(args: argparse.Namespace) -> None:
    """Manage session reminders."""
    from synapt.recall.reminders import (
        add_reminder,
        clear_reminder,
        load_reminders,
        pop_pending,
        format_for_session_start,
    )

    if args.pending:
        pending = pop_pending()  # Single load-save cycle
        if not pending:
            return  # Silent — no reminders (hook context)
        print(format_for_session_start(pending))
        return

    if args.list:
        reminders = load_reminders()
        if not reminders:
            print("No reminders.")
            return
        for r in reminders:
            sticky = " [sticky]" if r.sticky else ""
            shown = f" (shown {r.shown_count}x)" if r.shown_count > 0 else ""
            print(f"  {r.id}  {r.text}{sticky}{shown}")
        return

    if args.clear is not None:
        # --clear with optional ID (empty string means clear all)
        rid = args.clear if args.clear else None
        count = clear_reminder(rid)
        if count:
            print(f"Cleared {count} reminder(s).")
        else:
            print("No reminders to clear.")
        return

    # Default: add a reminder
    if not args.text:
        print("Usage: synapt recall remind \"text to remember\"", file=sys.stderr)
        sys.exit(1)

    reminder = add_reminder(args.text, sticky=args.sticky)
    sticky_label = " (sticky)" if args.sticky else ""
    print(f"Added reminder{sticky_label}: {reminder.text} (id: {reminder.id})")


def cmd_channel(args: argparse.Namespace) -> None:
    """Cross-worktree agent communication channels."""
    from synapt.recall.channel import (
        channel_join,
        channel_leave,
        channel_post,
        channel_read,
        channel_who,
        channel_heartbeat,
        channel_unread,
        channel_pin,
        channel_directive,
        channel_mute,
        channel_unmute,
        channel_kick,
        channel_broadcast,
        channel_list_channels,
    )

    action = args.action
    channel = args.channel or "dev"

    if action == "join":
        join_kwargs: dict = {"channel": channel}
        if getattr(args, "name", None):
            join_kwargs["display_name"] = args.name
        print(channel_join(**join_kwargs))
    elif action == "leave":
        print(channel_leave(channel=channel))
    elif action == "post":
        if not args.message:
            print("Usage: synapt recall channel post <channel> \"message\"", file=sys.stderr)
            sys.exit(1)
        print(channel_post(channel=channel, message=args.message, pin=args.pin))
    elif action == "read":
        read_kwargs: dict = {"channel": channel, "limit": args.limit, "since": args.since}
        if getattr(args, "detail", None):
            read_kwargs["detail"] = args.detail
        print(channel_read(**read_kwargs))
    elif action == "who":
        print(channel_who())
    elif action == "heartbeat":
        print(channel_heartbeat())
    elif action == "unread":
        counts = channel_unread()
        if not counts:
            print("No channel memberships -- join a channel first.")
        else:
            print("## Unread messages")
            for ch, count in sorted(counts.items()):
                marker = f" ({count} new)" if count > 0 else ""
                print(f"  #{ch}: {count}{marker}")
    elif action == "pin":
        if not args.message:
            print("Usage: synapt recall channel pin <channel> \"message_id\"", file=sys.stderr)
            sys.exit(1)
        print(channel_pin(channel=channel, message_id=args.message))
    elif action == "directive":
        if not args.message or not args.to:
            print("Usage: synapt recall channel directive <channel> \"message\" --to <agent>", file=sys.stderr)
            sys.exit(1)
        print(channel_directive(channel=channel, message=args.message, to=args.to))
    elif action == "mute":
        if not args.target:
            print("Usage: synapt recall channel mute <channel> --target <agent>", file=sys.stderr)
            sys.exit(1)
        print(channel_mute(target=args.target, channel=channel))
    elif action == "unmute":
        if not args.target:
            print("Usage: synapt recall channel unmute <channel> --target <agent>", file=sys.stderr)
            sys.exit(1)
        print(channel_unmute(target=args.target, channel=channel))
    elif action == "kick":
        if not args.target:
            print("Usage: synapt recall channel kick <channel> --target <agent>", file=sys.stderr)
            sys.exit(1)
        print(channel_kick(target=args.target, channel=channel))
    elif action == "broadcast":
        if not args.message:
            print("Usage: synapt recall channel broadcast \"message\"", file=sys.stderr)
            sys.exit(1)
        print(channel_broadcast(message=args.message))
    elif action == "list":
        channels = channel_list_channels()
        if not channels:
            print("No channels yet.")
        else:
            print("Channels: " + ", ".join(f"#{c}" for c in channels))
    elif action == "search":
        if not args.message:
            print("Usage: synapt recall channel search <channel> \"query\"", file=sys.stderr)
            sys.exit(1)
        from synapt.recall.channel import channel_search
        results = channel_search(args.message)
        if not results:
            print("No matching messages.")
        else:
            for r in results:
                ts = r["timestamp"][:16]
                print(f"  [{r['message_id']}] #{r['channel']} {ts}  {r['from']}: {r['body']}")
    elif action == "rename":
        if not args.message:
            print("Usage: synapt recall channel rename <channel> \"new name\"", file=sys.stderr)
            sys.exit(1)
        from synapt.recall.channel import channel_rename
        print(channel_rename(new_name=args.message))
    elif action == "chat":
        from synapt.recall.channel_chat import main as chat_main
        chat_main(
            channel=channel,
            name=args.target,  # reuse --target for --name
            poll=float(args.limit) if args.limit != 20 else 1.0,
        )


_GLOBAL_HOOKS: dict[str, dict[str, str | int]] = {
    "SessionStart": {
        "command": "synapt recall hook session-start",
        "timeout": 60,
        "matcher": "startup|resume|clear|fork",
    },
    "SessionEnd": {"command": "synapt recall checkpoint --event-json -", "timeout": 3},
    "PreCompact": {"command": "synapt recall hook precompact", "timeout": 300},
}


def _install_global_hooks() -> int:
    """Register synapt hooks in ~/.claude/settings.json.

    Returns number of hooks newly installed (0 if all already present).
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    hooks = settings.setdefault("hooks", {})
    installed = 0
    migrated = 0

    # Migrate renamed commands and the former unbounded default SessionEnd.
    # The heavy handler remains available explicitly, but must not coexist
    # with the bounded checkpoint in automatic hook configuration.
    _OLD_HOOK_PREFIX = "synapse recall hook "
    _OBSOLETE_COMMANDS = {"synapt recall hook session-end"}
    for event in list(hooks.keys()):
        matchers = hooks.get(event, [])
        for m in matchers:
            if not isinstance(m, dict):
                continue
            inner = m.get("hooks", [])
            filtered = [
                h for h in inner
                if not (
                    isinstance(h, dict)
                    and (
                        h.get("command", "").startswith(_OLD_HOOK_PREFIX)
                        or h.get("command") in _OBSOLETE_COMMANDS
                    )
                )
            ]
            if len(filtered) < len(inner):
                migrated += len(inner) - len(filtered)
            m["hooks"] = filtered

    for event, hook_cfg in _GLOBAL_HOOKS.items():
        command = hook_cfg["command"]
        timeout = hook_cfg.get("timeout", 60)
        desired_matcher = str(hook_cfg.get("matcher", ""))
        matchers = hooks.setdefault(event, [])

        # Move an existing command when its source matcher changed. In
        # particular, the old catch-all SessionStart must stop launching on a
        # compaction-triggered start.
        for matcher in matchers:
            if not isinstance(matcher, dict):
                continue
            matcher_value = str(matcher.get("matcher") or "")
            if matcher_value == desired_matcher:
                continue
            inner = matcher.get("hooks", [])
            filtered = [
                hook for hook in inner
                if not (isinstance(hook, dict) and hook.get("command") == command)
            ]
            if len(filtered) < len(inner):
                migrated += len(inner) - len(filtered)
            matcher["hooks"] = filtered

        # Check if our command is already registered under the right matcher.
        already = any(
            isinstance(m, dict)
            and str(m.get("matcher") or "") == desired_matcher
            and any(
                isinstance(h, dict) and h.get("command") == command
                for h in m.get("hooks", [])
            )
            for m in matchers
        )
        if already:
            continue

        # Find or create the event's required matcher.
        target = None
        for m in matchers:
            if (
                isinstance(m, dict)
                and str(m.get("matcher") or "") == desired_matcher
            ):
                target = m
                break
        entry = {"type": "command", "command": str(command), "timeout": int(timeout)}
        if target is None:
            matchers.append({"matcher": desired_matcher, "hooks": [entry]})
        else:
            target.setdefault("hooks", []).append(entry)
        installed += 1

    if installed or migrated:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(settings, settings_path)

    return installed


def _catchup_archive_and_journal(project: Path, transcript_dir: Path) -> None:
    """Ensure all transcripts are archived and journaled.

    Called at session-start to catch up after /clear or crash where
    session-end didn't fire. Archives new/grown transcripts and writes
    journal entries for any sessions not already journaled.

    This is separate from _archive_and_build's synthesize_journal_stubs
    because it uses auto_extract_entry (which reads the actual transcript
    file for richer extraction) rather than synthesizing from index chunks.
    """
    from synapt.recall.archive import archive_transcripts
    from synapt.recall.journal import (
        _journal_path,
        _read_all_session_ids,
        append_entry,
        auto_extract_entry,
    )

    # Archive transcripts (copies from Claude Code's dir to .synapt/recall/transcripts/)
    copied = archive_transcripts(project, transcript_dir)
    if copied:
        print(f"  Catch-up: archived {len(copied)} transcript(s)", file=sys.stderr)

    # Journal from the archive (not source) — the archive has the full pre-/clear
    # content, while the source may be truncated after /clear.
    archive_dir = project_archive_dir(project)
    # Ambient (None): auto-journal writes must land in the SAME session-consistent
    # store the wake reads and the manual `recall journal` verb writes. Writing to
    # the cwd store here while readers resolve ambiently would split auto- and
    # manual journals across two stores (dual-use wake fix).
    journal_path = _journal_path()
    existing_ids = _read_all_session_ids(journal_path)
    journaled = 0

    # Prefer archive files; fall back to source for anything not yet archived
    journal_files: dict[str, Path] = {}
    for f in sorted(transcript_dir.glob("*.jsonl")):
        journal_files[f.name] = f
    if archive_dir.is_dir():
        for f in sorted(archive_dir.glob("*.jsonl")):
            journal_files[f.name] = f  # Archive overrides source

    for src_file in sorted(journal_files.values(), key=lambda p: p.name):
        # Extract session_id from this transcript
        session_id = extract_session_id(src_file)
        if not session_id or session_id in existing_ids:
            continue

        # Write a journal entry for this un-journaled session
        entry = auto_extract_entry(transcript_path=str(src_file), cwd=str(project))
        if entry.has_content():
            append_entry(entry, journal_path)
            existing_ids.add(session_id)  # Prevent duplicates within this loop
            journaled += 1
            print(f"  Catch-up: journaled session {session_id[:8]}", file=sys.stderr)

    if journaled:
        print(f"  Catch-up: wrote {journaled} journal entry(ies)", file=sys.stderr)


def generate_startup_context(
    project: Path,
    *,
    include_continuity: bool = True,
    current_session_id: str | None = None,
) -> list[str]:
    """Generate startup context lines for any tool (Claude, Codex, etc.).

    Returns a list of context strings covering:
    - Branch-aware journal context
    - Open PR status
    - Recent journal entries
    - Knowledge nodes
    - Pending reminders
    - Pending contradictions
    - Channel unread summary
    - Pending directives

    This is the shared core used by both cmd_hook (Claude SessionStart)
    and cmd_startup (Codex / tool-agnostic startup). Side effects like
    background indexing, archiving, and enrichment are NOT included here;
    those belong in cmd_hook which runs inside Claude's hook lifecycle.
    """
    continuity_lines: list[str] = []
    lines: list[str] = []
    journal_lines: list[str] = []
    compaction_line: str | None = None
    latest_authored_journal_timestamp: str | None = None
    authored_entries: list = []

    # 1. Branch-aware context
    try:
        from synapt.recall.journal import _get_branch
        branch = _get_branch(str(project))
        if branch and branch not in ("main", "master"):
            from synapt.recall.journal import _read_all_entries, _journal_path
            all_entries = []
            # Ambient (None): the wake must read the SAME journal store the
            # session's own recall/journal verbs read -- workspace-aware via
            # GRIPSPACE_ROOT -- not the cwd. An explicit project=cwd suppresses
            # the env override (the deliberate export/import --path contract) and
            # reads a blank/wrong store when cwd != workspace (dual-use wake fix).
            # `project` still drives the branch legs below, which ARE cwd facts.
            jf = _journal_path()
            if jf.exists():
                all_entries.extend(_read_all_entries(jf))
            branch_entries = [e for e in all_entries if e.branch == branch]
            if branch_entries:
                latest = sorted(branch_entries, key=lambda e: e.timestamp)[-1]
                if latest.focus:
                    continuity_lines.append(f"Branch context ({branch}): {latest.focus}")
                    if latest.decisions:
                        continuity_lines.append(f"  Decisions: {'; '.join(latest.decisions[:3])}")
                    if latest.next_steps:
                        continuity_lines.append(f"  Next steps: {'; '.join(latest.next_steps[:3])}")
    except Exception:
        pass

    # 2. Open PR status for current branch
    try:
        from synapt.recall.journal import _get_branch
        branch = _get_branch(str(project))
        if branch and branch not in ("main", "master"):
            import subprocess as _sp
            pr_result = _sp.run(
                ["gh", "pr", "list", "--head", branch, "--state", "open",
                 "--json", "number,title,reviews,url", "--limit", "1"],
                capture_output=True, text=True, timeout=10,
            )
            if pr_result.returncode == 0 and pr_result.stdout.strip() not in ("", "[]"):
                import json as _json
                prs = _json.loads(pr_result.stdout)
                for pr in prs:
                    n_reviews = len(pr.get("reviews", []))
                    continuity_lines.append(f"Open PR: #{pr['number']} -- {pr['title']} ({n_reviews} review(s))")
    except Exception:
        pass

    # 3. Journal entries. The read is bounded, so its omissions are part of
    # the result rather than an invisible implementation detail.
    try:
        from synapt.recall.journal import _read_all_entries, _journal_path, _dedup_entries
        from synapt.recall.journal import format_for_session_start
        # Ambient (None): resolve the session-consistent journal store, not cwd
        # (see the branch-context leg above; dual-use wake fix).
        jf = _journal_path()
        if jf.exists():
            all_entries = _dedup_entries(_read_all_entries(jf))
            rich = [e for e in all_entries if e.has_rich_content()]
            rich.sort(key=lambda e: e.timestamp, reverse=True)
            shown_entries = rich[:_WAKE_JOURNAL_ENTRY_LIMIT]
            journal_lines.append(
                "Journal read: "
                + json.dumps(
                    {
                        "shown": len(shown_entries),
                        "withheld": len(rich) - len(shown_entries),
                        "oldest_shown_at": (
                            shown_entries[-1].timestamp if shown_entries else None
                        ),
                    },
                    separators=(",", ":"),
                )
            )
            # A files-only authored checkpoint still supersedes an older raw
            # transcript tail even though it is not rich enough to render.
            authored = [
                entry for entry in all_entries
                if not entry.auto and entry.has_content()
            ]
            authored.sort(key=lambda entry: entry.timestamp, reverse=True)
            authored_entries = list(authored)
            if authored:
                latest_authored_journal_timestamp = authored[0].timestamp
            for entry in shown_entries:
                journal_lines.append(format_for_session_start(entry))
    except Exception:
        pass

    # 4. Latest runtime-authored compaction handoff. This sidecar is refreshed
    # by transcript indexing, so startup remains O(1) in transcript size.
    try:
        from synapt.recall.compaction import (
            format_agent_compaction_directive,
            format_compaction_summary,
            latest_agent_compaction_directive,
            latest_compaction_summary,
        )
        agent_id = os.environ.get("SYNAPT_AGENT_ID")
        agent_name = (
            os.environ.get("SYNAPT_AGENT_NAME") or os.environ.get("AGENT_NAME")
        )
        summary = None
        directive = latest_agent_compaction_directive(project, agent_name)
        if agent_id and directive:
            compaction_line = format_agent_compaction_directive(directive)
        else:
            summary = latest_compaction_summary(project, agent_id=agent_id)
        if not compaction_line and summary:
            compaction_line = format_compaction_summary(summary)
    except Exception:
        pass

    # 5a. Unclean end: the newest PREVIOUS transcript has no journal within
    # the grace window and no SessionEnd checkpoint of its own. A crash runs
    # no SessionEnd, so the checkpoint on disk (if any) is some other
    # session's; without this block that other tail renders as the bridge and
    # nothing says hours of work have no record (measured 2026-08-31). The
    # block carries the crashed session's OWN bounded tail and leads the wake.
    try:
        from synapt.recall.resume import format_unclean_end, gather_unclean_end

        # Exclusion is a call-site guarantee: a caller that cannot name the
        # session that is starting must not publish this verdict, or the wake
        # reports itself (the generic `synapt recall startup` path, Atlas r1).
        found = None
        if current_session_id:
            found = gather_unclean_end(
                project,
                exclude_session_id=current_session_id,
                authored=authored_entries,
            )
        if found:
            tail = None
            try:
                from synapt.checkpoint import capture_checkpoint

                tail = capture_checkpoint({
                    "transcript_path": str(found.transcript_path),
                    "session_id": found.session_id,
                    "cwd": str(project),
                    "hook_event_name": "SessionStart",
                    "reason": "unclean-end",
                })
            except Exception:
                tail = None
            continuity_lines.insert(0, format_unclean_end(found, tail))
    except Exception:
        pass

    # 5. Raw SessionEnd recovery checkpoint, but only while it is newer than
    # the latest authored journal. Once an authored handoff catches up, it is
    # authoritative and the raw tail disappears from startup.
    try:
        from synapt.checkpoint import (
            format_checkpoint,
            is_newer_than,
            read_checkpoint,
        )
        checkpoint = read_checkpoint(project)
        if checkpoint and is_newer_than(checkpoint, latest_authored_journal_timestamp):
            continuity_lines.append(format_checkpoint(checkpoint))
    except Exception:
        pass

    # Raw current-facing evidence must survive the startup byte budget before
    # older authored history. The compaction handoff follows it, then journals.
    if compaction_line:
        continuity_lines.append(compaction_line)
    continuity_lines.extend(journal_lines)

    # 6. Knowledge nodes
    try:
        from synapt.recall.knowledge import read_nodes, format_knowledge_for_session_start
        kn_text = format_knowledge_for_session_start(read_nodes())
        if kn_text:
            lines.append(kn_text)
    except Exception:
        pass

    # 7. Pending reminders
    try:
        from synapt.recall.reminders import pop_pending, format_for_session_start as fmt_reminders
        pending = pop_pending()
        if pending:
            lines.append(fmt_reminders(pending))
    except Exception:
        pass

    # 8. Pending contradictions
    try:
        from synapt.recall.server import format_contradictions_for_session_start
        contradictions_text = format_contradictions_for_session_start()
        if contradictions_text:
            lines.append(contradictions_text)
    except Exception:
        pass

    # 9. Channel unread summary
    try:
        from synapt.recall.channel import channel_join, channel_unread, channel_read
        role = "agent" if os.environ.get("SYNAPT_AGENT_ID") else "human"
        channel_join("dev", role=role)
        counts = channel_unread()
        if counts:
            unread_parts = [f"#{ch}: {n}" for ch, n in sorted(counts.items()) if n > 0]
            if unread_parts:
                lines.append(f"Channel: {', '.join(unread_parts)} unread")
            total_unread = sum(counts.values())
            if total_unread > 0:
                # A quiet night reads five messages in full. A backlog reads
                # one line per message up to thirty, so the wake covers what
                # happened rather than the three newest posts (2026-08-31:
                # eleven unread, three rendered, the two verdicts that
                # mattered withheld).
                if total_unread > 5:
                    summary = channel_read(
                        "dev", limit=min(total_unread, 30), show_pins=False, detail="min",
                    )
                else:
                    summary = channel_read("dev", limit=total_unread, show_pins=False)
                if summary:
                    lines.append(f"\nRecent #dev messages:\n{summary}")
    except Exception:
        pass

    # 10. Pending directives
    try:
        from synapt.recall.channel import check_directives
        directives = check_directives()
        if directives:
            lines.append(f"\nPending directives:\n{directives}")
    except Exception:
        pass

    return (continuity_lines if include_continuity else []) + lines


def _gripspace_root(project: Path) -> Path:
    """Return the configured or nearest gripspace root, or project."""
    env_root = os.environ.get("GRIPSPACE_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        if (root / ".gitgrip").is_dir():
            return root

    root = project
    while root != root.parent:
        if (root / ".gitgrip").is_dir():
            return root
        root = root.parent
    return project


def _agent_startup_config(project: Path, agent_name: str) -> dict[str, object]:
    """Load the current agent's startup config from .gitgrip/agents.toml."""
    if not agent_name:
        return {}

    agents_toml = _gripspace_root(project) / ".gitgrip" / "agents.toml"
    if not agents_toml.exists():
        return {}

    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(agents_toml, "rb") as f:
            cfg = tomllib.load(f)
    except Exception:
        return {}

    agent_cfg = cfg.get("agents", {}).get(agent_name, {})
    spawn_cfg = cfg.get("spawn", {})
    if not isinstance(agent_cfg, dict):
        return {}

    merged: dict[str, object] = {}
    if isinstance(spawn_cfg, dict):
        merged["channel"] = spawn_cfg.get("channel")
    merged.update(agent_cfg)
    return merged


def _detect_startup_tool(agent_cfg: dict[str, object]) -> str:
    """Detect whether startup text is being read by Codex or Claude."""
    explicit = os.environ.get("SYNAPT_TOOL") or os.environ.get("AGENT_TOOL")
    if explicit:
        return explicit.strip().lower()

    if os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_CI"):
        return "codex"

    configured = agent_cfg.get("tool")
    if isinstance(configured, str) and configured.strip():
        return configured.strip().lower()

    return "claude"


def _dev_loop_activation_prompt(project: Path) -> str | None:
    """Return a runtime-appropriate session-start prompt."""
    agent_name = os.environ.get("AGENT_NAME", "")
    if not agent_name:
        return None

    agent_cfg = _agent_startup_config(project, agent_name)
    channel = (
        os.environ.get("SYNAPT_CHANNELS", "").split(",", 1)[0].strip()
        or str(agent_cfg.get("channel") or "dev")
    )
    tool = _detect_startup_tool(agent_cfg)

    if tool == "codex":
        return (
            f"SessionStart:startup context loaded: Agent {agent_name} — join #{channel} "
            f"once with recall_channel if channel context is needed. The dev-loop is "
            f"deprecated for Codex because Codex has no CronCreate or async self-wake; "
            f"do not emulate a monitoring loop with sleeps or periodic self-polling. "
            f"Use the startup context to choose the next concrete task."
        )

    # No resume is performed here, so the label does not claim one. The prior
    # text read "SessionStart:resume hook success", which announced a handoff
    # the hook never carried out -- a label is not a mechanism, and this one
    # survived truncation more reliably than the context it mislabelled.
    #
    # The monitoring loop is likewise removed rather than reworded: a
    # cadence-poll spends context re-reading unchanged state, and an agent
    # that waits to be prompted does not need one.
    # Claude CAN self-wake where Codex cannot, which is why this instruction
    # had to be retired deliberately instead of falling away on its own.
    return (
        f"SessionStart:startup context loaded: Agent {agent_name} — join "
        f"#{channel} once with recall_channel if channel context is needed. "
        f"Do NOT create a cron loop or poll on a cadence: the monitoring "
        f"loop is deprecated. Wait to be prompted, and notify your "
        f"coordinator when a task completes. Prefer doing needed work over "
        f"reporting that work exists."
    )


def cmd_startup(args: argparse.Namespace) -> None:
    """Generate startup context for any tool (Codex, Claude, etc.).

    Prints the same context that Claude gets via SessionStart hooks,
    enabling Codex and other tools to achieve startup parity.

    Usage:
        synapt recall startup              # context for cwd
        synapt recall startup --compact    # single-line summary
        synapt recall startup --json       # machine-readable output
    """
    project = Path.cwd().resolve()

    # Optional: compact journal before surfacing (same as SessionStart)
    try:
        from synapt.recall.journal import compact_journal
        compact_journal()
    except Exception:
        pass

    # The runtime that is starting names itself in its env (Claude Code:
    # CLAUDE_CODE_SESSION_ID, Codex: CODEX_THREAD_ID). Without it the wake
    # cannot exclude the live transcript and does not judge an unclean end.
    current_session_id = (
        os.environ.get("CLAUDE_CODE_SESSION_ID")
        or os.environ.get("CODEX_THREAD_ID")
        or os.environ.get("SYNAPT_SESSION_ID")
        or None
    )
    context_lines = generate_startup_context(project, current_session_id=current_session_id)

    if not context_lines:
        if getattr(args, "json", False):
            print("{}")
        return

    if getattr(args, "json", False):
        import json
        print(json.dumps({"context": "\n".join(context_lines)}, indent=2))
    elif getattr(args, "compact", False):
        # The compact form is embedded directly into a runtime prompt. Give it
        # the same source-aware byte budget as SessionStart before flattening;
        # flattening an unbounded context merely turns the overflow into one
        # enormous logical line and can bury the current assignment.
        from synapt.recall.session_start import render_wake, WAKE_BUDGET_BYTES

        rendered = render_wake(context_lines, source="startup")
        parts = [line.strip() for line in rendered.splitlines() if line.strip()]
        compact = " | ".join(parts)
        # ``render_wake`` budgets its newline-delimited bytes. Replacing every
        # newline with a three-byte separator can expand the final transport,
        # and ``print`` adds one more byte. Enforce the promise on the bytes the
        # runtime actually receives.
        if len(compact.encode("utf-8")) + 1 > WAKE_BUDGET_BYTES:
            suffix = " … (compact output clipped)"
            room = WAKE_BUDGET_BYTES - 1 - len(suffix.encode("utf-8"))
            compact = (
                compact.encode("utf-8")[:room]
                .decode("utf-8", errors="ignore")
                .rstrip()
                + suffix
            )
        print(compact)
    else:
        for line in context_lines:
            print(line)


def cmd_grep_intercept(args: argparse.Namespace) -> None:
    """Emit advisory recall context for a Claude Code PreToolUse event."""
    from synapt.integrations.grep_intercept import (
        GrepInterceptConfig,
        build_pretooluse_output,
    )

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(payload, dict):
        return
    # This command is a JSON hook protocol. Index-load diagnostics belong to
    # interactive commands, not to the hook's stderr channel. Logging handlers
    # may retain the original stderr object, so redirecting stderr is not enough.
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        output = build_pretooluse_output(
            payload,
            config=GrepInterceptConfig(
                enabled=True,
                timeout_ms=max(1, args.timeout_ms),
            ),
        )
    finally:
        logging.disable(previous_logging_disable)
    if output is not None:
        print(json.dumps(output, separators=(",", ":")))


def _session_start_continuity_allowed(source: str) -> bool:
    """Return whether SessionStart should inject recovery context.

    A compaction start is always a no-op because the runtime has already
    carried the live context forward.  The remaining sources are governed by
    the user's continuity policy.  ``automatic`` deliberately respects
    ``/clear`` as an intent boundary.
    """
    source = (source or "startup").strip().lower()
    if source == "compact":
        return False

    from synapt.recall.config import load_config

    mode = load_config().get_session_start_continuity()
    if mode == "off":
        return False
    if mode == "explicit":
        return source == "resume"
    if mode == "automatic":
        return source in {"startup", "resume", "fork"}
    return True


def _spawn_session_start_catchup(project: Path) -> bool:
    """Start the one detached maintenance worker, if transcripts exist."""
    import subprocess

    if not project_transcript_dirs(project):
        return False
    subprocess.Popen(
        [sys.executable, "-m", "synapt.recall.cli", "catchup"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


def cmd_hook(args: argparse.Namespace) -> None:
    """Versioned hook handler — replaces shell scripts.

    Called directly from Claude Code hooks config:
        "command": "synapt recall hook session-start"
    """
    # Read the hook payload (JSON on stdin). SessionStart carries ``source``:
    # "startup" | "resume" | "clear" | "compact" | "fork". It used to be
    # drained and discarded; it is the one field that says WHY we are waking.
    payload: dict = {}
    try:
        raw = sys.stdin.read()
        parsed = json.loads(raw) if raw and raw.strip() else {}
        payload = parsed if isinstance(parsed, dict) else {}
    except Exception:
        payload = {}

    event = args.event

    if event == "session-start":
        source = str(payload.get("source") or "startup").strip().lower()
        if source == "compact":
            # The runtime already carried the live context across compaction.
            # Do not resolve recall state, render a wake, log a run, or start
            # deferred resume maintenance for this SessionStart.
            return

    include_continuity = True
    if event == "session-start":
        include_continuity = _session_start_continuity_allowed(source)

    # Opt-out check
    if (project_data_dir() / "no-auto-capture").exists():
        return

    if event == "session-start":
        # INVARIANT: this branch does O(1) work in transcript and index size,
        # prints inside a fixed byte budget, and records its own run.
        #
        # Why (2026-08-25, Ref #856, #119): the archive + journal catch-up
        # that used to run here first was O(archive) — 5.5 minutes on one
        # machine — so the hook was killed at the 60s timeout and emitted
        # NOTHING. A timed-out hook's output is discarded, not truncated, and
        # no signal anywhere said so. Every O(n) job now lives in `catchup`,
        # spawned detached; this branch prints from existing state only.
        from synapt.recall.session_start import HookRun, render_wake

        source = str(payload.get("source") or "startup").strip().lower()
        run = HookRun("session-start", source)
        warning = run.begin()
        project = Path.cwd().resolve()
        banners: list[str] = []

        # 0. Stale MCP server warning — surface prominently so agent acts (#428)
        with run.phase("version_check"):
            try:
                from synapt.recall.server import _check_version_stale
                stale_warning = _check_version_stale()
                if stale_warning:
                    banners.append(
                        f"WARNING: {stale_warning}\n"
                        "Call recall_reload to restart the MCP server with the latest code."
                    )
            except Exception:
                pass

        # 1. ONE detached process for everything unbounded: archive, journal
        #    catch-up, journal compaction, incremental build, one enrich.
        #    Sequenced inside `catchup` under its own lock so they cannot
        #    fight each other (or a manual build) for the build lock.
        with run.phase("spawn_catchup"):
            try:
                _spawn_session_start_catchup(project)
            except Exception:
                banners.append("WARNING: could not spawn `synapt recall catchup`; index and journal will not update this session.")

        # 2. Compact journal (dedup + sort) before surfacing context. Cheap
        #    (tens of ms on a 5 MB journal) and it keeps the read consistent.
        with run.phase("compact_journal"):
            try:
                from synapt.recall.journal import compact_journal
                removed = compact_journal()
                if removed:
                    print(f"  Journal: compacted ({removed} duplicate(s) removed)", file=sys.stderr)
            except Exception:
                pass

        # 3. Surface startup context from EXISTING state (shared with
        #    cmd_startup for Codex parity) ...
        with run.phase("context"):
            lines = generate_startup_context(
                project,
                include_continuity=include_continuity,
                current_session_id=str(payload.get("session_id") or "") or None,
            )

        # 4. ... rendered inside the byte budget, head line first, full text
        #    on disk with a pointer. The harness previews ~2 KB of this.
        with run.phase("render"):
            text = render_wake(lines, source=source, run=run,
                               warning=warning, banners=banners)
        sys.stdout.write(text)

        # 5. Dev-loop activation prompt — deterministic hook replaces
        #    unreliable skill auto-activation (~20%). The agent reads this
        #    system reminder as its startup instruction; it does not start
        #    a monitoring loop.
        extra = 0
        try:
            prompt = _dev_loop_activation_prompt(project)
            if prompt:
                print(f"\n{prompt}")
                extra = len(prompt.encode("utf-8")) + 1
        except Exception:
            pass  # Loop activation is non-critical

        run.finish(output_bytes=len(text.encode("utf-8")) + extra)

    elif event == "session-end":
        # 1. Archive transcripts locally
        cmd_archive(argparse.Namespace())

        # 2. Write auto-extracted journal entry
        cmd_journal(argparse.Namespace(read=False, write=True, list=False, show=None,
                                       focus=None, done=None, decisions=None, next=None))

        # 3. Leave channel
        try:
            from synapt.recall.channel import channel_leave
            channel_leave("dev", reason="session ended")
        except Exception:
            pass  # Channel is non-critical

    elif event == "precompact":
        # Rebuild with sync
        project = Path.cwd().resolve()
        if project_transcript_dirs(project):
            final_index = _archive_and_build(project, use_embeddings=False, incremental=True)
            if final_index:
                stats = final_index.stats()
                print(f"synapt: rebuilt index ({stats['chunk_count']} chunks)", file=sys.stderr)
            _sync_after_rebuild(project)
            # Write an interim journal entry so mid-session state is captured
            # even when SessionEnd never fires (crash, kill, etc.).
            _precompact_journal_write(project)

        # Heartbeat to keep presence alive
        try:
            from synapt.recall.channel import channel_heartbeat
            channel_heartbeat()
        except Exception:
            pass  # Channel is non-critical

    elif event == "check-directives":
        # Fast path: check for unread directives targeted at this agent.
        # Output goes to stdout → appears as system reminder in context.
        # Empty output = invisible (no noise when nothing pending).
        try:
            from synapt.recall.channel import check_directives
            output = check_directives()
            if output:
                print(output)
        except Exception:
            pass  # Never block a tool call for channel issues


def _precompact_journal_write(project: Path) -> None:
    """Write an auto-extracted journal entry during PreCompact, if not already journaled.

    Unlike the recall_journal MCP tool, this intentionally writes file-list-only
    stubs (entries with only files_modified, no focus/done/decisions) — a file
    list is better than nothing for crash-recovery purposes when context compacts
    mid-session before the user writes a rich entry.

    Reduces the chance of duplicate entries — the dedup check prevents the
    most common case, but concurrent hook invocations (e.g. SessionEnd
    racing with PreCompact) can still produce duplicates that
    compact_journal will clean up on the next build.
    """
    transcript_path = latest_transcript_path(project)
    if not transcript_path:
        return

    session_id = extract_session_id(transcript_path)
    if not session_id:
        return

    # Ambient (None): the PreCompact auto-journal write lands in the same
    # session-consistent store the wake reads (dual-use wake fix; see
    # _catchup_archive_and_journal for the split-store rationale).
    journal_file = _journal_path()
    if session_id in _read_all_session_ids(journal_file):
        logger.debug("PreCompact journal skip — session %s already journaled", session_id[:8])
        return

    try:
        entry = auto_extract_entry(transcript_path=transcript_path, cwd=str(project))
        if entry and entry.has_content():
            # Re-check immediately before writing: auto_extract_entry runs git
            # subprocesses that can take several seconds, creating a wide TOCTOU
            # window where a concurrent SessionEnd hook may have written the entry.
            if session_id in _read_all_session_ids(journal_file):
                logger.debug("PreCompact journal skip (post-extract) — session %s already journaled",
                             session_id[:8])
                return
            append_entry(entry, journal_file)
            print(f"  Journal: interim entry written for session {session_id[:8]}",
                  file=sys.stderr)
    except Exception as exc:
        logger.warning("PreCompact journal write failed: %s", exc, exc_info=True)


def cmd_install_hook(args: argparse.Namespace) -> None:
    """Install global hooks (SessionStart, SessionEnd, PreCompact)."""
    installed = _install_global_hooks()
    if installed:
        print(f"Installed {installed} hook(s) in ~/.claude/settings.json")
    else:
        print("All hooks already registered in ~/.claude/settings.json")
    print("\nThe synapt index will auto-rebuild on context compaction.")


def cmd_setup(args: argparse.Namespace) -> None:
    """One-command setup: build index, register MCP server, install hook."""
    from synapt.recall.archive import (
        load_sync_config,
        save_sync_config,
        download_from_hf,
    )

    project = Path.cwd().resolve()
    print(f"[setup] Project: {project}")
    print()

    # Warn about legacy index
    legacy = _check_legacy_index()
    if legacy:
        print(f"[setup] Note: legacy index found at {legacy}")
        print("[setup] In-project indexes are now used. You can remove the old index with:")
        print(f"  rm -rf {legacy}")
        print()

    total_steps = 5 if not args.no_hook else 4
    step = 1

    # --- 0. Configure sync (if requested) ---
    sync_repo = None
    if args.sync:
        # Parse --sync hf:user/repo or just user/repo
        sync_arg = args.sync
        if sync_arg.startswith("hf:"):
            sync_repo = sync_arg[3:]
        else:
            sync_repo = sync_arg

        config = load_sync_config(project)
        existing_sync = config.get("sync", {})
        config["sync"] = {
            "provider": "hf",
            "repo_id": sync_repo,
            "auto_sync": True,
            "extra_files": existing_sync.get("extra_files", []),
        }
        save_sync_config(project, config)
        print(f"[setup] Sync configured: HuggingFace -> {sync_repo}")

        # Pull from HF first (new machine scenario)
        print(f"[setup] Pulling transcripts from HF ...")
        downloaded = download_from_hf(project, sync_repo)
        if downloaded:
            print(f"  Downloaded {downloaded} transcript(s) from HF")
        else:
            print(f"  No new transcripts from HF")
        print()

    # --- 1. Archive + build index ---
    print(f"[setup] Step {step}/{total_steps}: Archiving transcripts & building index ...")
    step += 1

    transcript_dir = project_transcript_dir(project)
    codex_count = 0
    try:
        from synapt.recall.codex import discover_codex_sessions, list_codex_transcripts
        codex_dir = discover_codex_sessions()
        if codex_dir:
            codex_count = len(list_codex_transcripts(codex_dir, project_dir=project))
            if codex_count:
                print(f"  Found {codex_count} Codex transcript files")
    except Exception:
        codex_count = 0
    if transcript_dir:
        jsonl_count = len(list(transcript_dir.glob("*.jsonl")))
        print(f"  Found {jsonl_count} transcript files at {transcript_dir}")

    # Also count any pre-existing archive transcripts (e.g., from HF pull)
    archive_dir = project_archive_dir(project)
    if archive_dir.exists():
        archive_count = len(list(archive_dir.glob("*.jsonl")))
        if archive_count:
            print(f"  Found {archive_count} archived transcript(s)")

    has_transcripts = bool(transcript_dir) or codex_count > 0 or (archive_dir.exists() and any(archive_dir.glob("*.jsonl")))
    stats = None

    if not has_transcripts:
        print(f"  No transcripts found yet. Skipping index build.")
        print(f"  Start a Claude Code or Codex session, then run 'synapt init' again to index.")
    else:
        use_emb = not args.no_embeddings
        if use_emb:
            print("  Computing embeddings ...")

        final_index = _archive_and_build(
            project,
            use_embeddings=use_emb,
        )

        if final_index and final_index.chunks:
            stats = final_index.stats()
            print(f"  Index saved: {stats['chunk_count']} chunks, {stats['session_count']} sessions")
            if stats.get("date_range"):
                dr = stats["date_range"]
                print(f"  Date range: {dr['earliest'][:10]} to {dr['latest'][:10]}")
        else:
            print(f"  No chunks parsed from transcripts. Index not built.")
    print()

    # --- 2. Register MCP server ---
    print(f"[setup] Step {step}/{total_steps}: Registering MCP server ...")
    step += 1
    scope = "user" if args.global_scope else "project"

    if not shutil.which("claude"):
        print("  Warning: 'claude' CLI not found in PATH. Skipping MCP registration.", file=sys.stderr)
        print("  Register manually: claude mcp add -s user -t stdio synapt synapt-server", file=sys.stderr)
    else:
        try:
            result = subprocess.run(
                ["claude", "mcp", "add", "-s", scope, "-t", "stdio",
                 "synapt", "synapt-server"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                print(f"  Registered MCP server (scope: {scope})")
            else:
                stderr = result.stderr.strip()
                if "already exists" in stderr.lower():
                    print(f"  MCP server already registered (scope: {scope})")
                else:
                    print(f"  MCP registration returned: {stderr or result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  Warning: MCP registration failed: {e}", file=sys.stderr)
    print()

    # --- 3. Install hooks ---
    if not args.no_hook:
        print(f"[setup] Step {step}/{total_steps}: Installing hooks ...")
        step += 1
        installed = _install_global_hooks()
        if installed:
            print(f"  Registered {installed} global hook(s) in ~/.claude/settings.json")
        else:
            print(f"  Global hooks already registered")
    else:
        print(f"[setup] Step {step}/{total_steps}: Skipping hooks (--no-hook)")
        step += 1
    print()

    # --- 4. Install Codex skill ---
    print(f"[setup] Step {step}/{total_steps}: Installing Codex skill ...")
    step += 1
    skill_path = _install_codex_skill()
    if skill_path is not None:
        print(f"  Installed dev-loop skill at {skill_path}")
    else:
        print("  No packaged Codex skill found")
    print()

    # --- 5. Ensure .gitignore ---
    _ensure_gitignore(project)

    # --- 5. Push to HF if sync configured ---
    if sync_repo:
        from synapt.recall.archive import upload_to_hf
        print("[setup] Pushing to HF ...")
        cfg = load_sync_config(project).get("sync", {})
        extra = cfg.get("extra_files", [])
        uploaded = upload_to_hf(project, sync_repo, extra_files=extra)
        if uploaded:
            print(f"  Uploaded {uploaded} file(s)")
        else:
            print(f"  All files already synced")
        print()

    # --- Summary ---
    print("=" * 50)
    print("  synapt setup complete!")
    index_dir = project_index_dir(project)
    if stats:
        total_size = sum(fp.stat().st_size for fp in index_dir.iterdir() if fp.is_file())
        print(f"  Index:    {index_dir} ({format_size(total_size)})")
        print(f"  Chunks:   {stats['chunk_count']}")
        print(f"  Sessions: {stats['session_count']}")
    else:
        print(f"  Index:    not yet built (no transcripts)")
    print(f"  MCP:      registered (scope: {scope})")
    if not args.no_hook:
        print(f"  Hook:     installed")
    if skill_path is not None:
        print(f"  Codex:    skill deployed ({skill_path})")
    if sync_repo:
        print(f"  Sync:     {sync_repo}")
    print()
    print("  Restart Claude Code to activate MCP tools.")
    print("=" * 50)


def cmd_sync(args: argparse.Namespace) -> None:
    """Sync transcripts to/from HuggingFace."""
    from synapt.recall.archive import (
        load_sync_config,
        upload_to_hf,
        download_from_hf,
    )

    project = Path.cwd().resolve()
    config = load_sync_config(project)
    sync = config.get("sync", {})

    repo_id = args.repo or sync.get("repo_id")
    if not repo_id:
        print("Error: no sync target configured.", file=sys.stderr)
        print("Run 'synapt setup --sync hf:user/repo' first,", file=sys.stderr)
        print("or specify --repo user/repo.", file=sys.stderr)
        sys.exit(1)

    direction = args.direction or "both"

    if direction in ("pull", "both"):
        print(f"[sync] Pulling from {repo_id} ...")
        downloaded = download_from_hf(project, repo_id)
        print(f"  Downloaded {downloaded} transcript(s)")

    if direction in ("push", "both"):
        print(f"[sync] Pushing to {repo_id} ...")
        extra = sync.get("extra_files", [])
        uploaded = upload_to_hf(project, repo_id, extra_files=extra)
        print(f"  Uploaded {uploaded} file(s)")


# ---------------------------------------------------------------------------
# HuggingFace download (legacy — used by build --hf)
# ---------------------------------------------------------------------------

def _download_hf_transcripts(repo_id: str) -> Path | None:
    """Download transcript files from HuggingFace into the project archive."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. pip install synapt[hf]", file=sys.stderr)
        return None

    token = os.environ.get("HF_TOKEN")

    # Download into project archive instead of global dir
    archive_dir = project_archive_dir()
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hf] Listing files in {repo_id} ...")
    api = HfApi(token=token)
    try:
        repo_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    except Exception as e:
        print(f"Error listing HF repo: {e}", file=sys.stderr)
        return None

    jsonl_files = [f for f in repo_files if f.endswith(".jsonl") and "transcripts/" in f]
    print(f"[hf] Found {len(jsonl_files)} transcript files")

    for fname in jsonl_files:
        basename = fname.split("/")[-1]
        local_path = archive_dir / basename
        if local_path.exists():
            print(f"  {basename}: already downloaded")
            continue
        print(f"  Downloading {basename} ...")
        try:
            hf_hub_download(
                repo_id, fname,
                repo_type="dataset",
                local_dir=str(archive_dir.parent),
                token=token,
            )
        except Exception as e:
            print(f"  Error: {e}")

    return archive_dir


# ---------------------------------------------------------------------------
# migrate-channels helpers (exposed for testing)
# ---------------------------------------------------------------------------

def _resolve_org_id_for_cli(project_dir: "Path | None") -> "str | None":
    """Thin wrapper so tests can patch org resolution independently."""
    from synapt.recall.channel import _resolve_org_id
    return _resolve_org_id(project_dir)


def _resolve_project_id_for_cli(project_dir: "Path | None") -> "str | None":
    """Thin wrapper so tests can patch project resolution independently."""
    from synapt.recall.channel import _resolve_project_id
    return _resolve_project_id(project_dir)


def _get_global_channels_dir(project_dir: "Path | None" = None) -> "Path":
    """Return the global channels root (~/.synapt/channels/).

    Exposed as a standalone function so tests can patch it without touching
    Path.home() or channel internals.
    """
    from pathlib import Path as _Path
    return _Path.home() / ".synapt" / "channels"


def cmd_migrate_channels(args: "argparse.Namespace") -> None:
    """Migrate local .synapt/recall/channels/ to global ~/.synapt/channels/ store."""
    from pathlib import Path as _Path
    from synapt.recall.channel import migrate_channels_to_global, _local_channels_dir

    project_dir = _Path(args.project_dir) if args.project_dir else None

    org_id = args.org or _resolve_org_id_for_cli(project_dir)
    project_id = args.project or _resolve_project_id_for_cli(project_dir)

    if not org_id:
        import sys as _sys
        print(
            "Error: --org is required (or run from a gripspace with a manifest URL)",
            file=_sys.stderr,
        )
        _sys.exit(1)
    if not project_id:
        import sys as _sys
        print(
            "Error: --project is required (or run from a gripspace with a manifest URL)",
            file=_sys.stderr,
        )
        _sys.exit(1)

    local_dir = _local_channels_dir(project_dir)
    global_dir = _get_global_channels_dir(project_dir)

    if not local_dir.exists():
        print(f"No local channels found at {local_dir}")
        return

    n_before = len(list(local_dir.glob("*.jsonl")))
    migrate_channels_to_global(local_dir, global_dir, org_id, project_id)
    target = global_dir / org_id / project_id
    print(f"Migrated {n_before} channel(s) → {target}")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def cmd_maintain(args: argparse.Namespace) -> None:
    """Grind LLM cluster summaries on request, bounded, and report the backlog.

    This is the grinder's explicit home. `build` no longer calls it, because an
    unbounded grind inside a build made a routine rebuild unpredictably slow and
    gave the operator no way to decline it.

    The backlog is REPORTED, never silently drained: a queue that is being
    worked and a queue that is stuck look identical from outside unless the
    number left is printed, so it is printed even when it is zero.
    """
    # Import the MODULE, not the name. The grinder is swapped at runtime -- by
    # tests, and by anyone pointing it at a different backend -- and a
    # `from ... import upgrade_large_cluster_summaries` binds at import time, so
    # the swap would be silently ignored while everything still looked correct.
    from synapt.recall import clustering
    from synapt.recall.core import project_data_dir
    from synapt.recall.storage import RecallDB

    min_chunks = 5
    index_dir = project_data_dir(None) / "index"
    if not index_dir.exists():
        print(f"No recall index at {index_dir}; run `synapt build` first.")
        return

    from synapt.recall.sharding import is_sharded
    if is_sharded(index_dir):
        from synapt.recall.sharded_db import ShardedRecallDB
        db = ShardedRecallDB.open(index_dir)
    else:
        db = RecallDB(index_dir / "recall.db")

    try:
        upgraded = clustering.upgrade_large_cluster_summaries(
            db, min_chunks=min_chunks, max_upgrades=args.limit
        )
        # Count what is LEFT with the grinder's own eligibility criteria rather
        # than a paraphrase of them: a backlog measured by a slightly different
        # query is a number about a different question.
        remaining = db._conn.execute(
            "SELECT COUNT(*) "
            "FROM clusters c "
            "LEFT JOIN cluster_summaries cs "
            "  ON c.cluster_id = cs.cluster_id AND cs.method = 'llm' "
            "WHERE c.status = 'active' "
            "  AND c.chunk_count >= ? "
            "  AND cs.cluster_id IS NULL",
            (min_chunks,),
        ).fetchone()[0]
    finally:
        db.close()

    print(f"maintain: upgraded {upgraded} cluster summar{'y' if upgraded == 1 else 'ies'}")
    print(f"  {remaining} remaining above the {min_chunks}-chunk threshold")


def cmd_catchup(args: argparse.Namespace) -> None:
    """Everything the session-start hook defers, in one detached process.

    1. Archive + journal catch-up for every transcript dir (O(archive))
    2. Journal compaction
    3. ``build --incremental`` (O(index))
    4. ``enrich --max-entries 1``

    Sequenced, not parallel: the old hook spawned build and enrich as two
    separate processes and ran the catch-up inline, so three things could
    contend for the build lock and the hook itself could be killed mid-way.
    Single-flight under ``catchup.lock``: a second catchup while one is
    running (two sessions starting together) steps aside and says so on
    stderr, because two of these at once would double-journal and then
    queue on the build lock for a minute each.
    """
    from synapt.recall.journal import compact_journal

    project = Path.cwd().resolve()
    data_dir = project_data_dir(project)
    fd = _acquire_build_lock(data_dir, timeout=0, name="catchup.lock")
    if fd is None:
        print(f"[catchup] already running ({_build_lock_busy_message(data_dir, name='catchup.lock')}); yielding",
              file=sys.stderr)
        return
    try:
        dirs = project_transcript_dirs(project)
        for transcript_dir in dirs:
            _catchup_archive_and_journal(project, transcript_dir)
        removed = compact_journal()
        if removed:
            print(f"[catchup] journal: compacted ({removed} duplicate(s) removed)", file=sys.stderr)
        if getattr(args, "no_build", False) or not dirs:
            return
        subprocess.run(
            [sys.executable, "-m", "synapt.recall.cli", "build", "--incremental"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, "-m", "synapt.recall.cli", "enrich", "--max-entries", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    finally:
        _release_build_lock(fd)


class _FullRebuild(argparse.Action):
    """``--full`` sets BOTH ``full`` and ``incremental`` rather than leaving one
    to be derived from the other.

    A reader of the parsed namespace should not have to compute
    ``incremental = not full``: two fields that must agree are two chances to
    disagree, and this pair has already diverged once across two surfaces.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        namespace.full = True
        namespace.incremental = False


def make_parser() -> argparse.ArgumentParser:
    """Build the full `synapt` argument parser.

    Extracted from ``main()`` so DEFAULTS ARE TESTABLE. While the parser was
    built inline, no test could ask what a bare ``synapt build`` actually
    does, which is how the CLI and the MCP surface drifted to opposite
    defaults for the same operation without anything going red.
    """
    parser = argparse.ArgumentParser(
        prog="synapt",
        description="Persistent conversational memory for Claude Code sessions (per-project)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Setup
    setup_parser = subparsers.add_parser("setup", help="One-command setup: build index, register MCP, install hook")
    setup_parser.add_argument("--global", dest="global_scope", action="store_true", help="Register MCP server globally (default: project-scoped)")
    setup_parser.add_argument("--no-embeddings", action="store_true", help="Skip embeddings (BM25-only, faster build)")
    setup_parser.add_argument("--no-hook", action="store_true", help="Skip global hook installation")
    setup_parser.add_argument("--sync", default=None, help="Configure HF sync (e.g., hf:user/repo or user/repo)")

    # Build
    build_parser = subparsers.add_parser("build", help="Build transcript index for current project")
    build_parser.add_argument("--source", action="append", help="Directory with .jsonl transcript files (can specify multiple). Auto-discovers if omitted.")
    build_parser.add_argument("--hf", help="HuggingFace repo ID (e.g., user/dataset-name)")
    build_parser.add_argument("--chatgpt-archive", help="Path to ChatGPT export .zip (or conversations.json)")
    build_parser.add_argument("--out", default=None, help="Output directory for index (default: per-project)")
    build_parser.add_argument("--no-embeddings", action="store_true", help="Skip embeddings (BM25-only, faster build)")
    # DEFAULT: incremental. `--full` is the explicit opt-out.
    #
    # The CLI defaulted to a FULL rebuild while MCP `recall_build` defaulted to
    # incremental: the same operation with opposite defaults depending on which
    # surface you reached through. Nothing went red because the parser was built
    # inline inside main(), so no test could ask what a bare `synapt build` does.
    # test_cli_and_mcp_defaults_agree now pins the two together; changing either
    # default alone fails it.
    build_parser.set_defaults(incremental=True, full=False)
    _build_mode = build_parser.add_mutually_exclusive_group()
    _build_mode.add_argument(
        "--full", action=_FullRebuild, nargs=0,
        help="Rebuild everything from scratch (opt out of the incremental default)",
    )
    _build_mode.add_argument(
        "--incremental", action="store_true",
        help="Skip already-indexed files. Now the default; still accepted so "
             "scripts and hooks that pass it explicitly keep working.",
    )
    build_parser.add_argument("--rescrub", action="store_true", help="Re-scrub archived transcripts with latest patterns before building")

    # Split
    split_parser = subparsers.add_parser(
        "split",
        help="Split monolithic recall.db into quarterly shards (experimental)",
    )
    split_parser.add_argument("--dry-run", action="store_true", help="Show split plan without writing")

    # Search
    search_parser = subparsers.add_parser("search", help="Search transcript index")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--index", default=None, help="Index directory (default: per-project)")
    search_parser.add_argument("--max-chunks", type=int, default=5, help="Max chunks to return")
    search_parser.add_argument("--max-tokens", type=int, default=None, help="Token budget (default: from config, or 500)")
    search_parser.add_argument("--max-sessions", type=int, default=None, help="Progressive: only search N recent sessions")
    search_parser.add_argument("--after", default=None, help="Only results after this date (ISO 8601, e.g. 2026-02-28)")
    search_parser.add_argument("--before", default=None, help="Only results before this date (ISO 8601, e.g. 2026-03-01)")
    search_parser.add_argument("--profile", action="store_true", help="Show per-phase timing breakdown")

    # Benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Run search pipeline benchmarks")
    benchmark_parser.add_argument("--index", default=None, help="Index directory (default: per-project)")
    benchmark_parser.add_argument("--json", dest="json_output", action="store_true", help="Output results as JSON")
    benchmark_parser.add_argument("--queries", default=None, help="Semicolon-separated queries (default: built-in set)")
    benchmark_parser.add_argument("--iterations", type=int, default=5, help="Iterations per query (default: 5)")

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.add_argument("--index", default=None, help="Index directory (default: per-project)")

    # Sessions
    sessions_parser = subparsers.add_parser("sessions", help="List recent sessions with summaries")
    sessions_parser.add_argument("--index", default=None, help="Index directory (default: per-project)")
    sessions_parser.add_argument("--max-sessions", type=int, default=20, help="Max sessions to list (default: 20)")
    sessions_parser.add_argument("--after", default=None, help="Only sessions after this date (ISO 8601)")
    sessions_parser.add_argument("--before", default=None, help="Only sessions before this date (ISO 8601)")

    # Resume (session tail — pick up after an unclean stop)
    resume_parser = subparsers.add_parser(
        "resume", help="Show the tail of the most recent session (pick up where it stopped)"
    )
    resume_parser.add_argument("session", nargs="?", default=None,
                               help="Session id or unique prefix (default: newest session)")
    resume_parser.add_argument("--index", default=None, help="Index directory (default: per-project)")
    resume_parser.add_argument("--turns", type=int, default=10,
                               help="How many trailing turns to show (default: 10)")

    # Rebuild (hook-triggered)
    rebuild_parser = subparsers.add_parser("rebuild", help="Incremental rebuild (for hooks)")
    rebuild_parser.add_argument("--out", default=None, help="Output directory (default: per-project)")
    rebuild_parser.add_argument("--sync", action="store_true", help="Push new transcripts to HF after rebuild")
    rebuild_parser.add_argument("--enrich", type=int, nargs="?", const=1, default=0,
                                metavar="N", help="Enrich up to N auto-stubs after rebuild (default: 1)")

    # Sync
    sync_parser = subparsers.add_parser("sync", help="Sync transcripts to/from HuggingFace")
    sync_parser.add_argument("direction", nargs="?", choices=["push", "pull", "both"], default="both", help="Sync direction (default: both)")
    sync_parser.add_argument("--repo", default=None, help="HuggingFace repo ID (overrides config)")

    # Archive (lightweight local copy, no indexing)
    subparsers.add_parser("archive", help="Archive transcripts locally (no indexing)")

    # Export / Import (portable recall snapshot)
    export_parser = subparsers.add_parser("export", help="Export portable recall data to a .synapt-archive file")
    export_parser.add_argument("output", nargs="?", default=None, help="Output .synapt-archive path (default: <project>.synapt-archive)")
    export_parser.add_argument("--exclude-transcripts", action="store_true", help="Skip raw transcript archives")
    export_parser.add_argument("--exclude-channels", action="store_true", help="Skip channel history files")
    export_parser.add_argument("--path", default=None, help="Workspace root to export (default: SYNAPT_RECALL_ROOT, else GRIPSPACE_ROOT, else inferred from git/gripspace, else the current directory; $HOME is refused, not used)")

    import_parser = subparsers.add_parser("import", help="Import portable recall data from a .synapt-archive file")
    import_parser.add_argument("archive", help="Path to a .synapt-archive file")
    import_mode = import_parser.add_mutually_exclusive_group()
    import_mode.add_argument("--merge", action="store_true", help="Merge imported data into existing recall state")
    import_mode.add_argument("--replace", action="store_true", help="Replace existing recall state (default)")
    import_parser.add_argument("--path", default=None, help="Workspace root to import into (default: SYNAPT_RECALL_ROOT, else GRIPSPACE_ROOT, else inferred from git/gripspace, else the current directory; $HOME is refused, not used)")

    # Transcript (display/save a session)
    transcript_parser = subparsers.add_parser("transcript", help="Display or save a session transcript")
    transcript_parser.add_argument("session_id", nargs="?", default=None, help="Session ID (default: most recent)")
    transcript_parser.add_argument("--save", action="store_true", help="Archive transcript locally")
    transcript_parser.add_argument("--list", action="store_true", help="List available sessions")

    # Journal
    journal_parser = subparsers.add_parser("journal", help="Session journal — structured session logging")
    journal_parser.add_argument("--read", action="store_true", help="Print latest entry's next steps (for hooks)")
    journal_parser.add_argument("--write", action="store_true", help="Write a journal entry (auto-extracts context)")
    journal_parser.add_argument("--list", action="store_true", help="List recent journal entries")
    journal_parser.add_argument("--show", type=int, default=None, help="Show Nth most recent entry")
    journal_parser.add_argument("--focus", default=None, help="What this session was about")
    journal_parser.add_argument("--done", default=None, help="What got done (one per line; a single-line value falls back to semicolon-separated)")
    journal_parser.add_argument("--decisions", default=None, help="Key decisions (one per line; a single-line value falls back to semicolon-separated)")
    journal_parser.add_argument("--next", default=None, help="Next steps (one per line; a single-line value falls back to semicolon-separated)")
    journal_parser.add_argument("--repair", action="store_true",
                                help="Recover fields swallowed by an unclosed tool-call parameter (append-only)")
    journal_parser.add_argument("--dry-run", action="store_true",
                                help="With --repair: report what would change, write nothing")
    journal_parser.add_argument("--all-stores", action="store_true",
                                help="With --repair: sweep every worktree journal, not just this one")
    journal_parser.add_argument("--path", default=None,
                                help="With --repair: target this store explicitly (or this data root "
                                     "with --all-stores) instead of resolving from the working directory")

    # Enrich
    enrich_parser = subparsers.add_parser("enrich", help="Enrich auto-journal stubs using MLX (local LLM)")
    enrich_parser.add_argument("--model", default="mlx-community/Ministral-3-3B-Instruct-2512-4bit",
                               help="MLX model to use (default: Ministral-3-3B-Instruct-2512-4bit)")
    enrich_parser.add_argument("--adapter-path", default="",
                               help="LoRA adapter path for enrichment")
    enrich_parser.add_argument("--dry-run", action="store_true", help="Show what would be enriched without doing it")
    enrich_parser.add_argument("--max-entries", type=int, default=0, help="Max entries to enrich (0 = unlimited)")
    enrich_parser.add_argument("--init-from", metavar="TRANSCRIPT",
                               help="Bootstrap journal from a large transcript file (segments by time gaps)")
    enrich_parser.add_argument("--gap-minutes", type=int, default=60,
                               help="Minimum gap in minutes between segments (default: 60, used with --init-from)")

    # Consolidate (memory consolidation — "sleep")
    consolidate_parser = subparsers.add_parser(
        "consolidate", aliases=["sleep"],
        help="Extract durable knowledge from journal entries",
    )
    consolidate_parser.add_argument("--model", default="mlx-community/Ministral-3-3B-Instruct-2512-4bit",
                                     help="MLX model to use")
    consolidate_parser.add_argument("--dry-run", action="store_true",
                                     help="Show what would be consolidated without doing it")
    consolidate_parser.add_argument("--force", action="store_true",
                                     help="Reprocess all entries, ignoring last consolidation timestamp")
    consolidate_parser.add_argument("--min-entries", type=int, default=3,
                                     help="Minimum enriched entries to trigger consolidation (default: 3)")
    consolidate_parser.add_argument("--show", action="store_true",
                                     help="Show existing knowledge nodes")
    consolidate_parser.add_argument("--adapter-path", default="",
                                     help="LoRA adapter path for knowledge extraction")

    # Remind
    remind_parser = subparsers.add_parser("remind", help="Manage session reminders")
    remind_parser.add_argument("text", nargs="?", default=None, help="Reminder text to add")
    remind_parser.add_argument("--sticky", action="store_true", help="Keep reminder across sessions")
    remind_parser.add_argument("--list", action="store_true", help="List all reminders")
    remind_parser.add_argument("--clear", nargs="?", const="", default=None, help="Clear reminder by ID (or all if no ID)")
    remind_parser.add_argument("--pending", action="store_true", help="Show and mark pending reminders (for hooks)")

    # Startup (tool-agnostic startup context — Codex parity with Claude SessionStart)
    startup_parser = subparsers.add_parser(
        "startup",
        help="Generate startup context (journal, reminders, channel) for any tool",
    )
    startup_parser.add_argument("--json", action="store_true", dest="json",
                                help="Output as JSON")
    startup_parser.add_argument("--compact", action="store_true",
                                help="Single-line summary for prompt injection")

    grep_intercept_parser = subparsers.add_parser(
        "grep-intercept",
        help="Add bounded recall context to grep-shaped PreToolUse events",
    )
    grep_intercept_parser.add_argument(
        "--timeout-ms",
        type=int,
        default=500,
        help="Inner recall deadline in milliseconds (default: 500)",
    )

    checkpoint_parser = subparsers.add_parser(
        "checkpoint",
        help="Capture a bounded SessionEnd recovery checkpoint",
    )
    checkpoint_parser.add_argument(
        "--event-json", default="-",
        help="Hook event JSON path, or - for stdin",
    )

    # Hook (versioned hook commands — called directly from Claude Code hooks config)
    hook_parser = subparsers.add_parser("hook", help="Run a Claude Code hook (session-start, session-end, precompact, check-directives)")
    hook_parser.add_argument("event", choices=["session-start", "session-end", "precompact", "check-directives"],
                             help="Hook event to handle")

    # Install hook (legacy — kept for backward compat)
    subparsers.add_parser("install-hook", help="Install global hooks (SessionStart, SessionEnd, PreCompact)")

    rescrub_parser = subparsers.add_parser("rescrub", help="Re-scrub archived transcripts with latest secret patterns")
    rescrub_parser.add_argument("--no-rebuild", action="store_true", help="Scrub transcripts but skip index rebuild")
    rescrub_parser.add_argument("--no-embeddings", action="store_true", help="Skip embeddings during rebuild")

    # Channel (cross-worktree communication)
    channel_parser = subparsers.add_parser("channel", help="Cross-worktree agent communication channels")
    channel_parser.add_argument("action",
                                choices=["post", "read", "who", "join", "leave",
                                         "heartbeat", "unread", "pin",
                                         "directive", "mute", "unmute", "kick",
                                         "broadcast", "list", "search", "chat"],
                                help="Channel action")
    channel_parser.add_argument("channel", nargs="?", default="dev",
                                help="Channel name (default: dev)")
    channel_parser.add_argument("message", nargs="?", default=None,
                                help="Message body (for post/pin/directive/broadcast)")
    channel_parser.add_argument("--limit", type=int, default=20,
                                help="Max messages to return (for 'read' action, default: 20)")
    channel_parser.add_argument("--since", default=None,
                                help="Only messages after this ISO timestamp (for 'read' action)")
    channel_parser.add_argument("--pin", action="store_true",
                                help="Also pin the message (for 'post' action)")
    channel_parser.add_argument("--to", default=None,
                                help="Target agent for 'directive' action")
    channel_parser.add_argument("--target", default=None,
                                help="Agent to mute/unmute/kick (id, display name, or griptree)")
    channel_parser.add_argument("--detail", default=None,
                                choices=["max", "high", "medium", "low", "min"],
                                help="Output detail level for read (default: medium)")
    channel_parser.add_argument("--name", default=None,
                                help="Display name for join action")

    catchup_parser = subparsers.add_parser(
        "catchup",
        help="Run the session-start hook's deferred maintenance: archive, journal "
             "catch-up, compaction, incremental build, one enrich. The hook spawns "
             "this detached; run it by hand to catch up now.",
    )
    catchup_parser.add_argument(
        "--no-build", action="store_true",
        help="Archive and journal only; skip the incremental build and enrich",
    )

    maintain_parser = subparsers.add_parser(
        "maintain",
        help="Upgrade cluster summaries with an LLM, bounded, and report the backlog",
    )
    maintain_parser.add_argument(
        "--limit", type=int, default=5,
        help="Maximum summaries to generate this run (default: 5). Bounded by "
             "default on purpose: an unbounded grind is what this command exists "
             "to replace.",
    )

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate local .synapt/recall/channels/ to global ~/.synapt/channels/ store",
    )
    migrate_parser.add_argument(
        "--project-dir", default=None,
        help="Path to the gripspace root (default: current directory)",
    )
    migrate_parser.add_argument(
        "--org", default=None,
        help="Org ID (auto-detected from gripspace manifest if not set)",
    )
    migrate_parser.add_argument(
        "--project", default=None,
        help="Project ID (auto-detected from gripspace manifest if not set)",
    )
    return parser


def main():
    # Configure logging so build progress is visible on stderr.
    # Only set up if no handlers exist yet (avoid duplicate output when
    # called from the MCP server, which configures its own logging).
    if not logging.getLogger("synapt").handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stderr,
        )

    parser = make_parser()

    args = parser.parse_args()

    if args.command == "maintain":
        cmd_maintain(args)
    elif args.command == "catchup":
        cmd_catchup(args)
    elif args.command == "setup":
        cmd_setup(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "sessions":
        cmd_sessions(args)
    elif args.command == "resume":
        cmd_resume(args)
    elif args.command == "rebuild":
        cmd_rebuild(args)
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "archive":
        cmd_archive(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "transcript":
        cmd_transcript(args)
    elif args.command == "journal":
        cmd_journal(args)
    elif args.command == "enrich":
        cmd_enrich(args)
    elif args.command in ("consolidate", "sleep"):
        cmd_consolidate(args)
    elif args.command == "remind":
        cmd_remind(args)
    elif args.command == "startup":
        cmd_startup(args)
    elif args.command == "grep-intercept":
        cmd_grep_intercept(args)
    elif args.command == "checkpoint":
        from synapt.checkpoint import main as checkpoint_main
        raise SystemExit(checkpoint_main(["--event-json", args.event_json]))
    elif args.command == "hook":
        cmd_hook(args)
    elif args.command == "install-hook":
        cmd_install_hook(args)
    elif args.command == "channel":
        cmd_channel(args)
    elif args.command == "rescrub":
        cmd_rescrub(args)
    elif args.command == "migrate":
        cmd_migrate_channels(args)


if __name__ == "__main__":
    main()
