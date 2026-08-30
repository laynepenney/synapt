"""ShardedRecallDB — tree-structured storage wrapper for recall.

Migration-path step 1: wraps a single RecallDB and adds shard-awareness.
Initially delegates everything to the underlying monolithic DB. Future PRs
will route chunk queries to per-quarter data shards.

Usage:
    # Drop-in replacement for RecallDB
    db = ShardedRecallDB.open(index_dir)
    db.fts_search("query")  # searches all shards
    db.close()

See issue #89 for the full design.
"""

from __future__ import annotations

import heapq
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

from synapt.recall.sharding import is_sharded, list_shards
from synapt.recall.storage import RecallDB


class ShardedRecallDB:
    """Shard-aware wrapper around RecallDB.

    Phase 1 (this PR): delegates to a single RecallDB. The ``open()``
    factory detects whether the index directory uses the sharded layout
    (index.db + data_*.db) or the monolithic layout (recall.db) and
    opens accordingly.

    Phase 2 (follow-up): routes chunk FTS queries to per-quarter data
    shards, merges results, and keeps index.db for knowledge/clusters.

    Chunk-facing methods that rely on rowids must be explicitly overridden
    here in sharded mode. Falling through to ``index.db`` via ``__getattr__``
    is only safe for index-only surfaces (knowledge, clusters, metadata,
    access tracking).
    """

    def __init__(self, index_db: RecallDB, data_dbs: list[RecallDB] | None = None):
        self._index = index_db
        self._data_dbs = data_dbs or []

    @staticmethod
    def _encode_chunk_rowid(shard_idx: int, rowid: int) -> int:
        """Encode a shard-qualified rowid into one globally unique integer."""
        return (shard_idx << 32) | rowid

    @staticmethod
    def _decode_chunk_rowid(encoded_rowid: int) -> tuple[int, int]:
        """Decode a global rowid back into ``(shard_idx, local_rowid)``."""
        return encoded_rowid >> 32, encoded_rowid & 0xFFFFFFFF

    def _iter_data_shards(self) -> list[tuple[int, RecallDB]]:
        """Return ``(shard_idx, db)`` pairs for chunk-bearing databases."""
        if self._data_dbs:
            return list(enumerate(self._data_dbs, start=1))
        return [(0, self._index)]

    @staticmethod
    def _overlay_rowid(rowid: int) -> int:
        return -rowid

    def _overlay_ids(self) -> set[str]:
        return set(self._index.query_tail_id_rowid_map())

    def _suppressed_base_sessions(self) -> set[str]:
        return self._index.query_tail_suppressed_sessions()

    def _base_chunk_visible(self, chunk) -> bool:  # noqa: ANN001
        return chunk.session_id not in self._suppressed_base_sessions()

    def _merge_overlay_chunks(self, chunks):  # noqa: ANN001, ANN201
        overlay = self._index.load_query_tail_chunks()
        overlay_ids = {chunk.id for chunk in overlay}
        suppressed = self._suppressed_base_sessions()
        return [
            chunk
            for chunk in chunks
            if chunk.id not in overlay_ids and chunk.session_id not in suppressed
        ] + overlay

    def _group_encoded_rowids(self, rowids: list[int]) -> dict[int, list[int]]:
        """Group global rowids by shard index, preserving local rowids."""
        grouped: dict[int, list[int]] = {}
        for rowid in rowids:
            shard_idx, local_rowid = self._decode_chunk_rowid(rowid)
            grouped.setdefault(shard_idx, []).append(local_rowid)
        return grouped

    @classmethod
    def open(cls, index_dir: Path) -> ShardedRecallDB:
        """Open a sharded or monolithic recall database.

        Auto-detects layout:
          - If ``index.db`` exists → sharded (index + data shards)
          - If ``recall.db`` exists → monolithic (single DB wraps both)
          - Otherwise → creates new monolithic ``recall.db``
        """
        if is_sharded(index_dir):
            index_db = RecallDB(index_dir / "index.db")
            data_dbs = [RecallDB(p) for p in list_shards(index_dir)]
            return cls(index_db, data_dbs)

        # Monolithic: single recall.db serves as both index and data
        db = RecallDB(index_dir / "recall.db")
        return cls(db, [])

    @classmethod
    def open_readonly(cls, index_dir: Path) -> ShardedRecallDB:
        """Open an existing layout without DDL, migrations, or write access."""
        if is_sharded(index_dir):
            index_db = RecallDB.open_readonly(index_dir / "index.db")
            try:
                data_dbs = [RecallDB.open_readonly(p) for p in list_shards(index_dir)]
            except Exception:
                index_db.close()
                raise
            return cls(index_db, data_dbs)

        return cls(RecallDB.open_readonly(index_dir / "recall.db"), [])

    # -- Delegated methods (index DB) --------------------------------------

    def load_manifest(self) -> dict:
        return self._index.load_manifest()

    def save_manifest(self, manifest: dict) -> None:
        self._index.save_manifest(manifest)

    def load_knowledge_nodes(self, status: str = "active") -> list[dict]:
        return self._index.load_knowledge_nodes(status)

    def save_knowledge_nodes(self, nodes: list[dict]) -> None:
        self._index.save_knowledge_nodes(nodes)

    def upsert_knowledge_node(self, node: dict) -> None:
        self._index.upsert_knowledge_node(node)

    def knowledge_fts_search(self, query: str, limit: int = 20,
                             include_historical: bool = False) -> list[tuple]:
        return self._index.knowledge_fts_search(query, limit, include_historical)

    def knowledge_by_rowid(self, rowids: list[int]) -> dict[int, dict]:
        return self._index.knowledge_by_rowid(rowids)

    def get_knowledge_node(self, node_id: str) -> dict | None:
        return self._index.get_knowledge_node(node_id)

    def list_pending_contradictions(self) -> list[dict]:
        return self._index.list_pending_contradictions()

    def add_pending_contradiction(self, **kwargs) -> int:
        return self._index.add_pending_contradiction(**kwargs)

    def resolve_contradiction(self, contradiction_id: int, status: str = "confirmed") -> bool:
        return self._index.resolve_contradiction(contradiction_id, status)

    def has_pending_contradiction_for(self, old_node_id: str) -> bool:
        return self._index.has_pending_contradiction_for(old_node_id)

    def pending_contradiction_count(self) -> int:
        return self._index.pending_contradiction_count()

    def save_clusters(self, clusters: list[dict], memberships: list[tuple]) -> None:
        self._index.save_clusters(clusters, memberships)

    def save_cluster_summary(self, cluster_id: str, summary: str, **kwargs) -> None:
        self._index.save_cluster_summary(cluster_id, summary, **kwargs)

    # -- Chunk methods (data shards or monolithic) -------------------------

    def load_chunks(self) -> list["TranscriptChunk"]:  # noqa: F821
        """Load chunks from all data shards, or from monolithic DB."""
        if self._data_dbs:
            all_chunks = []
            for db in self._data_dbs:
                all_chunks.extend(db.load_chunks())
            return self._merge_overlay_chunks(all_chunks)
        return self._merge_overlay_chunks(self._index.load_chunks())

    def session_indexed_extent(self, session_id: str) -> dict | None:
        """Return the furthest base-only byte extent across chunk shards."""
        extents = [
            extent
            for _, db in self._iter_data_shards()
            if (extent := db.session_indexed_extent(session_id)) is not None
        ]
        if not extents:
            return None
        return max(extents, key=lambda item: item["observed_complete_offset"])

    def load_chunk_headers(self) -> list["TranscriptChunk"]:  # noqa: F821
        """Load lightweight chunk metadata from all shards."""
        if self._data_dbs:
            all_chunks = []
            for db in self._data_dbs:
                all_chunks.extend(db.load_chunk_headers())
            return self._merge_overlay_chunks(all_chunks)
        return self._merge_overlay_chunks(self._index.load_chunk_headers())

    def load_chunk_by_rowid(self, rowid: int):  # noqa: ANN201
        """Load one chunk by shard-qualified rowid."""
        if rowid < 0:
            return self._index.load_query_tail_chunk_by_rowid(-rowid)
        if self._data_dbs:
            shard_idx, local_rowid = self._decode_chunk_rowid(rowid)
            if shard_idx <= 0 or shard_idx > len(self._data_dbs):
                return None
            chunk = self._data_dbs[shard_idx - 1].load_chunk_by_rowid(local_rowid)
        else:
            chunk = self._index.load_chunk_by_rowid(rowid)
        if chunk is not None and not self._base_chunk_visible(chunk):
            return None
        return chunk

    def load_chunks_by_rowids(self, rowids: list[int]):  # noqa: ANN201
        """Load multiple chunks keyed by shard-qualified rowids."""
        overlay_rowids = [-rowid for rowid in rowids if rowid < 0]
        overlay = {
            -rowid: chunk
            for rowid, chunk in self._index.load_query_tail_chunks_by_rowids(
                overlay_rowids
            ).items()
        }
        base_rowids = [rowid for rowid in rowids if rowid >= 0]
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            loaded = dict(overlay)
            for shard_idx, local_rowids in self._group_encoded_rowids(base_rowids).items():
                if shard_idx <= 0 or shard_idx > len(self._data_dbs):
                    continue
                partial = self._data_dbs[shard_idx - 1].load_chunks_by_rowids(local_rowids)
                loaded.update({
                    self._encode_chunk_rowid(shard_idx, local_rowid): chunk
                    for local_rowid, chunk in partial.items()
                })
            return {
                rowid: chunk
                for rowid, chunk in loaded.items()
                if rowid < 0 or chunk.session_id not in suppressed
            }
        loaded = self._index.load_chunks_by_rowids(base_rowids)
        loaded.update(overlay)
        return {
            rowid: chunk
            for rowid, chunk in loaded.items()
            if rowid < 0 or chunk.session_id not in suppressed
        }

    def session_overview(self) -> dict[str, dict]:
        """Return merged session metadata across all chunk shards."""
        result: dict[str, dict] = {}
        suppressed = self._suppressed_base_sessions()
        for _, db in self._iter_data_shards():
            for session_id, overview in db.session_overview().items():
                if session_id in suppressed:
                    continue
                current = result.get(session_id)
                if current is None:
                    result[session_id] = dict(overview)
                    continue
                if overview["has_real_activity"] and not current["has_real_activity"]:
                    current["activity"] = overview["activity"]
                elif overview["has_real_activity"] == current["has_real_activity"]:
                    current["activity"] = max(
                        current["activity"], overview["activity"]
                    )
                current["has_real_activity"] = (
                    current["has_real_activity"] or overview["has_real_activity"]
                )
                if overview["earliest_ts"]:
                    current["earliest_ts"] = min(
                        filter(None, (current["earliest_ts"], overview["earliest_ts"]))
                    )
                if overview["latest_ts"]:
                    current["latest_ts"] = max(
                        current["latest_ts"], overview["latest_ts"]
                    )
                current["turn_count"] += overview["turn_count"]
                if not current.get("transcript_path") and overview.get("transcript_path"):
                    current["transcript_path"] = overview["transcript_path"]
        for chunk in self._index.load_query_tail_chunks():
            current = result.setdefault(
                chunk.session_id,
                {
                    "activity": (0, ""),
                    "earliest_ts": "",
                    "latest_ts": "",
                    "turn_count": 0,
                    "has_real_activity": False,
                    "transcript_path": chunk.transcript_path,
                },
            )
            current["earliest_ts"] = min(
                filter(None, (current["earliest_ts"], chunk.timestamp))
            ) if current["earliest_ts"] or chunk.timestamp else ""
            current["latest_ts"] = max(current["latest_ts"], chunk.timestamp)
            current["turn_count"] += int(chunk.turn_index >= 0)
            current["has_real_activity"] = True
            current["activity"] = (0, current["latest_ts"])
        return result

    def session_activity(self) -> dict[str, tuple[int, str]]:
        """Return newest activity per session across all chunk shards."""
        return {
            session_id: overview["activity"]
            for session_id, overview in self.session_overview().items()
        }

    def load_session_chunks(self, session_id: str):  # noqa: ANN201
        """Load one session across all shards, oldest turn first."""
        return self.load_session_chunks_many(
            [session_id], include_journal=False
        ).get(session_id, [])

    def load_session_chunks_many(
        self,
        session_ids: list[str],
        include_journal: bool = True,
    ):  # noqa: ANN201
        """Load a bounded set of sessions across all shards."""
        grouped = {session_id: [] for session_id in session_ids}
        suppressed = self._suppressed_base_sessions()
        for _, db in self._iter_data_shards():
            partial = db.load_session_chunks_many(session_ids, include_journal)
            for session_id, chunks in partial.items():
                if session_id in suppressed:
                    continue
                grouped.setdefault(session_id, []).extend(chunks)
        overlay = self._index.load_query_tail_chunks()
        overlay_ids = {chunk.id for chunk in overlay}
        for session_id, chunks in grouped.items():
            grouped[session_id] = [
                chunk for chunk in chunks if chunk.id not in overlay_ids
            ]
        for chunk in overlay:
            if chunk.session_id in grouped:
                grouped[chunk.session_id].append(chunk)
        for chunks in grouped.values():
            chunks.sort(key=lambda chunk: chunk.turn_index)
        return grouped

    def load_session_listing(self, session_ids: list[str]) -> dict[str, list[dict]]:
        """Load summary fields for a bounded set of sessions across shards."""
        grouped: dict[str, list[dict]] = {session_id: [] for session_id in session_ids}
        suppressed = self._suppressed_base_sessions()
        for _, db in self._iter_data_shards():
            partial = db.load_session_listing(session_ids)
            for session_id, rows in partial.items():
                if session_id in suppressed:
                    continue
                grouped.setdefault(session_id, []).extend(rows)
        overlay = self._index.load_query_tail_chunks()
        overlay_ids = {chunk.id for chunk in overlay}
        for session_id, rows in grouped.items():
            grouped[session_id] = [
                row for row in rows if row.get("id") not in overlay_ids
            ]
        for chunk in overlay:
            if chunk.session_id not in grouped:
                continue
            grouped[chunk.session_id].append(
                {
                    "id": chunk.id,
                    "turn_index": chunk.turn_index,
                    "user_text": chunk.user_text,
                    "files_touched": chunk.files_touched,
                    "transcript_path": chunk.transcript_path,
                }
            )
        for rows in grouped.values():
            rows.sort(key=lambda row: row["turn_index"])
        return grouped

    def sample_chunk_texts(self, limit: int = 100) -> list[str]:
        """Return representative chunk text samples across all shards."""
        if self._data_dbs:
            if limit <= 0:
                return []
            total = self.chunk_count()
            if total == 0:
                return []
            samples: list[str] = []
            remaining = limit
            shard_info = [
                (shard_idx, db, db.chunk_count())
                for shard_idx, db in self._iter_data_shards()
            ]
            non_empty = [(shard_idx, db, count) for shard_idx, db, count in shard_info if count > 0]
            for pos, (_, db, shard_total) in enumerate(non_empty):
                target = max(1, round(limit * shard_total / total))
                non_empty_left = len(non_empty) - pos
                target = min(target, max(1, remaining - max(non_empty_left - 1, 0)))
                samples.extend(db.sample_chunk_texts(limit=target))
                remaining = max(limit - len(samples), 0)
                if remaining <= 0:
                    break
            return samples[:limit]
        return self._index.sample_chunk_texts(limit=limit)

    def chunk_count(self) -> int:
        """Number of chunks across all data shards, or monolithic DB."""
        suppressed = self._suppressed_base_sessions()
        overlay_refs = self._index.query_tail_chunk_refs()
        visible_overlay_ids = {
            chunk_id
            for chunk_id, session_id in overlay_refs
            if session_id not in suppressed
        }
        base_count = 0
        for _, db in self._iter_data_shards():
            base_count += db.chunk_count()
            base_count -= db.chunk_count_for_values("session_id", suppressed)
            base_count -= db.chunk_count_for_values("id", visible_overlay_ids)
        return base_count + self._index.query_tail_chunk_count()

    def content_hash(self) -> str:
        """Hash chunk content across all shards in global timestamp order.

        Mirrors ``RecallDB.content_hash()`` semantics so sharded and monolithic
        indexes produce the same invalidation signal for identical content.
        """
        if not self._data_dbs:
            return self._index.content_hash()

        import hashlib

        def _rows(db: RecallDB):
            return db._conn.execute(
                "SELECT timestamp, rowid, id, user_text, assistant_text, tool_content "
                "FROM chunks ORDER BY timestamp DESC, rowid DESC"
            )

        h = hashlib.sha256()
        merged = heapq.merge(
            *(_rows(db) for db in self._data_dbs),
            key=lambda row: (row[0], row[1]),
            reverse=True,
        )
        for _, _, chunk_id, user_text, assistant_text, tool_content in merged:
            h.update(
                f"{chunk_id}|{user_text or ''}|{assistant_text or ''}|{tool_content or ''}\n"
                .encode()
            )
        return h.hexdigest()[:16]

    def chunk_session_map(self) -> dict[int, str]:
        """Return a global ``{rowid: session_id}`` mapping for all chunks."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[int, str] = {}
            for shard_idx, db in self._iter_data_shards():
                for rowid, session_id in db.chunk_session_map().items():
                    if session_id in suppressed:
                        continue
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = session_id
            return result
        return {
            rowid: session_id
            for rowid, session_id in self._index.chunk_session_map().items()
            if session_id not in suppressed
        }

    def chunk_id_map(self) -> dict[int, str]:
        """Return a global ``{rowid: chunk_id}`` mapping for all chunks."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[int, str] = {}
            for shard_idx, db in self._iter_data_shards():
                sessions = db.chunk_session_map()
                for rowid, chunk_id in db.chunk_id_map().items():
                    if sessions.get(rowid) in suppressed:
                        continue
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = chunk_id
            return result
        sessions = self._index.chunk_session_map()
        return {
            rowid: chunk_id
            for rowid, chunk_id in self._index.chunk_id_map().items()
            if sessions.get(rowid) not in suppressed
        }

    def get_chunk_id_rowid_map(self) -> dict[str, int]:
        """Return ``{chunk_id: global_rowid}`` across all shards."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[str, int] = {}
            for shard_idx, db in self._iter_data_shards():
                sessions = db.chunk_session_map()
                for chunk_id, rowid in db.get_chunk_id_rowid_map().items():
                    if sessions.get(rowid) in suppressed:
                        continue
                    result[chunk_id] = self._encode_chunk_rowid(shard_idx, rowid)
            result.update({
                chunk_id: self._overlay_rowid(rowid)
                for chunk_id, rowid in self._index.query_tail_id_rowid_map().items()
            })
            return result
        sessions = self._index.chunk_session_map()
        result = {
            chunk_id: rowid
            for chunk_id, rowid in self._index.get_chunk_id_rowid_map().items()
            if sessions.get(rowid) not in suppressed
        }
        result.update({
            chunk_id: self._overlay_rowid(rowid)
            for chunk_id, rowid in self._index.query_tail_id_rowid_map().items()
        })
        return result

    def save_chunks(self, chunks: list["TranscriptChunk"]) -> None:  # noqa: F821
        """Save chunks to the appropriate database.

        In monolithic mode, delegates directly to the single DB.
        In sharded mode, closes all existing shards, deletes the old
        shard files, and re-distributes chunks across fresh shards
        using SHARD_CHUNK_THRESHOLD. This prevents the unbounded shard
        accumulation bug where every rebuild created a new full-copy
        shard (#sharding-bug-sprint-13).
        """
        if not self._data_dbs:
            self._index.save_chunks(chunks)
            return

        from synapt.recall.sharding import (
            SHARD_CHUNK_THRESHOLD,
            SHARD_METADATA_SQL,
            _update_shard_metadata,
            list_shards,
            shard_name_for_index,
        )

        index_dir = self._index._path.parent

        # Step 1: Close all existing data shard connections
        for db in self._data_dbs:
            try:
                db.close()
            except Exception:
                pass
        self._data_dbs = []

        # Step 2: Delete all old shard files (including WAL/SHM journals)
        for old_shard in list_shards(index_dir):
            for suffix in ("", "-wal", "-shm"):
                p = old_shard.parent / (old_shard.name + suffix)
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to delete %s", p)

        # Step 3: Clear stale shard_metadata in index.db
        try:
            self._index._conn.execute("DELETE FROM shard_metadata")
            self._index._conn.commit()
        except Exception:
            logger.warning("Failed to clear shard_metadata", exc_info=True)

        # Step 4: Sort chunks by timestamp for time-range sharding
        sorted_chunks = sorted(chunks, key=lambda c: c.timestamp or "")

        # Step 5: Distribute chunks across fresh shards
        shard_idx = 1
        for offset in range(0, len(sorted_chunks), SHARD_CHUNK_THRESHOLD):
            batch = sorted_chunks[offset:offset + SHARD_CHUNK_THRESHOLD]
            shard_path = index_dir / shard_name_for_index(shard_idx)
            shard_db = RecallDB(shard_path)
            shard_db.save_chunks(batch)
            self._data_dbs.append(shard_db)

            # Record shard metadata
            min_ts = batch[0].timestamp if batch else ""
            max_ts = batch[-1].timestamp if batch else ""
            is_last = offset + SHARD_CHUNK_THRESHOLD >= len(sorted_chunks)
            try:
                self._index._conn.execute(SHARD_METADATA_SQL)
                _update_shard_metadata(
                    self._index._conn, shard_path, len(batch),
                    min_ts or "", max_ts or "",
                    is_active=is_last,
                )
                self._index._conn.commit()
            except Exception:
                logger.warning("Failed to update shard metadata for %s",
                               shard_path.name, exc_info=True)
            shard_idx += 1

        logger.info("save_chunks: distributed %d chunks across %d shard(s)",
                     len(chunks), len(self._data_dbs))

    def retire_absorbed_query_tails(self) -> None:
        """Retire overlays only when the rebuilt base proves matching coverage."""
        from synapt.recall.storage import query_tail_source_key

        for cursor in self._index.load_query_tail_cursors():
            path = Path(cursor["transcript_path"])
            try:
                current_key = query_tail_source_key(cursor["session_id"], path)
            except OSError:
                continue
            if current_key != cursor["source_key"]:
                continue
            extent = self.session_indexed_extent(cursor["session_id"])
            if extent is None:
                continue
            if (
                extent["observed_complete_offset"]
                < cursor["observed_complete_offset"]
                or extent.get("latest_projected_timestamp", "")
                != cursor.get("latest_projected_timestamp", "")
            ):
                continue
            self._index.clear_query_tail(cursor["source_key"])

    def fts_search(self, query: str, limit: int = 100, **kwargs) -> list[tuple]:
        """FTS search across all shards, merging results by score.
        """
        overlay_ids = self._overlay_ids()
        suppressed = self._suppressed_base_sessions()
        overlay_results = [
            (self._overlay_rowid(rowid), score)
            for rowid, score in self._index.query_tail_fts_search(query, limit).copy()
        ]
        if self._data_dbs:
            all_results = list(overlay_results)
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search(query, limit=limit, **kwargs)
                chunk_ids = db.chunk_id_map()
                sessions = db.chunk_session_map()
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                    if chunk_ids.get(rowid) not in overlay_ids
                    and sessions.get(rowid) not in suppressed
                )
            # Sort by score descending (score is element [1])
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        chunk_ids = self._index.chunk_id_map()
        sessions = self._index.chunk_session_map()
        base = [
            (rowid, score)
            for rowid, score in self._index.fts_search(query, limit=limit, **kwargs)
            if chunk_ids.get(rowid) not in overlay_ids
            and sessions.get(rowid) not in suppressed
        ]
        merged = base + overlay_results
        merged.sort(key=lambda row: row[1], reverse=True)
        return merged[:limit]

    def fts_search_raw(self, fts_query: str, limit: int = 100) -> list[tuple[int, float]]:
        """Execute a pre-built FTS query across all shards."""
        overlay_ids = self._overlay_ids()
        suppressed = self._suppressed_base_sessions()
        overlay = [
            (self._overlay_rowid(rowid), score)
            for rowid, score in self._index.query_tail_fts_search_raw(
                fts_query, limit
            )
        ]
        if self._data_dbs:
            all_results = list(overlay)
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search_raw(fts_query, limit=limit)
                chunk_ids = db.chunk_id_map()
                sessions = db.chunk_session_map()
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                    if chunk_ids.get(rowid) not in overlay_ids
                    and sessions.get(rowid) not in suppressed
                )
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        chunk_ids = self._index.chunk_id_map()
        sessions = self._index.chunk_session_map()
        base = [
            (rowid, score)
            for rowid, score in self._index.fts_search_raw(fts_query, limit=limit)
            if chunk_ids.get(rowid) not in overlay_ids
            and sessions.get(rowid) not in suppressed
        ]
        merged = base + overlay
        merged.sort(key=lambda row: row[1], reverse=True)
        return merged[:limit]

    def fts_search_by_session(
        self,
        query: str,
        session_ids: list[str],
        limit: int = 100,
    ) -> list[tuple[int, float]]:
        """Session-scoped FTS search across all data shards."""
        overlay_ids = self._overlay_ids()
        suppressed = self._suppressed_base_sessions()
        overlay = [
            (self._overlay_rowid(rowid), score)
            for rowid, score in self._index.query_tail_fts_search_by_session(
                query, session_ids, limit
            )
        ]
        if self._data_dbs:
            all_results = list(overlay)
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search_by_session(query, session_ids, limit=limit)
                chunk_ids = db.chunk_id_map()
                sessions = db.chunk_session_map()
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                    if chunk_ids.get(rowid) not in overlay_ids
                    and sessions.get(rowid) not in suppressed
                )
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        chunk_ids = self._index.chunk_id_map()
        sessions = self._index.chunk_session_map()
        base = [
            (rowid, score)
            for rowid, score in self._index.fts_search_by_session(
                query, session_ids, limit=limit
            )
            if chunk_ids.get(rowid) not in overlay_ids
            and sessions.get(rowid) not in suppressed
        ]
        merged = base + overlay
        merged.sort(key=lambda row: row[1], reverse=True)
        return merged[:limit]

    def get_embeddings(self, rowids: list[int]) -> dict[int, list[float]]:
        """Fetch chunk embeddings keyed by global sharded rowids."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[int, list[float]] = {}
            for shard_idx, local_rowids in self._group_encoded_rowids(rowids).items():
                db = self._data_dbs[shard_idx - 1]
                sessions = db.chunk_session_map()
                for rowid, emb in db.get_embeddings(local_rowids).items():
                    if sessions.get(rowid) in suppressed:
                        continue
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = emb
            return result
        sessions = self._index.chunk_session_map()
        return {
            rowid: embedding
            for rowid, embedding in self._index.get_embeddings(rowids).items()
            if sessions.get(rowid) not in suppressed
        }

    def save_embeddings(self, embeddings: dict[int, list[float]]) -> None:
        """Store chunk embeddings keyed by global sharded rowids."""
        if self._data_dbs:
            grouped: dict[int, dict[int, list[float]]] = {}
            for rowid, emb in embeddings.items():
                shard_idx, local_rowid = self._decode_chunk_rowid(rowid)
                grouped.setdefault(shard_idx, {})[local_rowid] = emb
            for shard_idx, shard_embs in grouped.items():
                self._data_dbs[shard_idx - 1].save_embeddings(shard_embs)
            return
        self._index.save_embeddings(embeddings)

    def has_embeddings(self) -> bool:
        """True if any chunk-bearing database has stored embeddings."""
        if self._data_dbs:
            return any(db.has_embeddings() for db in self._data_dbs)
        return self._index.has_embeddings()

    def get_all_embeddings(self) -> dict[int, list[float]]:
        """Load all chunk embeddings keyed by global sharded rowids."""
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            result: dict[int, list[float]] = {}
            for shard_idx, db in self._iter_data_shards():
                sessions = db.chunk_session_map()
                for rowid, emb in db.get_all_embeddings().items():
                    if sessions.get(rowid) in suppressed:
                        continue
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = emb
            return result
        sessions = self._index.chunk_session_map()
        return {
            rowid: embedding
            for rowid, embedding in self._index.get_all_embeddings().items()
            if sessions.get(rowid) not in suppressed
        }

    def get_all_embeddings_numpy(self) -> "tuple[np.ndarray, list[int]]":
        """Load chunk embeddings into a numpy matrix with global rowids."""
        import numpy as np
        suppressed = self._suppressed_base_sessions()
        if self._data_dbs:
            matrices, all_rowids = [], []
            for shard_idx, db in self._iter_data_shards():
                mat, rids = db.get_all_embeddings_numpy()
                sessions = db.chunk_session_map()
                keep = [
                    pos
                    for pos, rowid in enumerate(rids)
                    if sessions.get(rowid) not in suppressed
                ]
                if keep:
                    matrices.append(mat[keep])
                    all_rowids.extend(
                        self._encode_chunk_rowid(shard_idx, rids[pos])
                        for pos in keep
                    )
            if not matrices:
                return np.empty((0, 384), dtype=np.float32), []
            return np.vstack(matrices), all_rowids
        matrix, rowids = self._index.get_all_embeddings_numpy()
        sessions = self._index.chunk_session_map()
        keep = [
            pos
            for pos, rowid in enumerate(rowids)
            if sessions.get(rowid) not in suppressed
        ]
        if not keep:
            return np.empty((0, 384), dtype=np.float32), []
        return matrix[keep], [rowids[pos] for pos in keep]

    # -- Access tracking (always index DB) ---------------------------------

    def record_access(self, *args, **kwargs) -> None:
        self._index.record_access(*args, **kwargs)

    # -- Passthrough for any method not explicitly wrapped ------------------

    def __getattr__(self, name: str):
        """Fall through to index DB for any unhandled method."""
        return getattr(self._index, name)

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close all database connections."""
        for db in self._data_dbs:
            db.close()
        self._index.close()

    @property
    def shard_count(self) -> int:
        """Number of data shards (0 for monolithic)."""
        return len(self._data_dbs)

    @property
    def is_monolithic(self) -> bool:
        """True if using a single recall.db (no shards)."""
        return not self._data_dbs
