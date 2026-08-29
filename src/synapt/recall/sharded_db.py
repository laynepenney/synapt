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
            return all_chunks
        return self._index.load_chunks()

    def load_chunk_headers(self) -> list["TranscriptChunk"]:  # noqa: F821
        """Load lightweight chunk metadata from all shards."""
        if self._data_dbs:
            all_chunks = []
            for db in self._data_dbs:
                all_chunks.extend(db.load_chunk_headers())
            return all_chunks
        return self._index.load_chunk_headers()

    def load_chunk_by_rowid(self, rowid: int):  # noqa: ANN201
        """Load one chunk by shard-qualified rowid."""
        if self._data_dbs:
            shard_idx, local_rowid = self._decode_chunk_rowid(rowid)
            if shard_idx <= 0 or shard_idx > len(self._data_dbs):
                return None
            return self._data_dbs[shard_idx - 1].load_chunk_by_rowid(local_rowid)
        return self._index.load_chunk_by_rowid(rowid)

    def load_chunks_by_rowids(self, rowids: list[int]):  # noqa: ANN201
        """Load multiple chunks keyed by shard-qualified rowids."""
        if self._data_dbs:
            loaded = {}
            for shard_idx, local_rowids in self._group_encoded_rowids(rowids).items():
                if shard_idx <= 0 or shard_idx > len(self._data_dbs):
                    continue
                partial = self._data_dbs[shard_idx - 1].load_chunks_by_rowids(local_rowids)
                loaded.update({
                    self._encode_chunk_rowid(shard_idx, local_rowid): chunk
                    for local_rowid, chunk in partial.items()
                })
            return loaded
        return self._index.load_chunks_by_rowids(rowids)

    def session_overview(self) -> dict[str, dict]:
        """Return merged session metadata across all chunk shards."""
        result: dict[str, dict] = {}
        for _, db in self._iter_data_shards():
            for session_id, overview in db.session_overview().items():
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
        for _, db in self._iter_data_shards():
            partial = db.load_session_chunks_many(session_ids, include_journal)
            for session_id, chunks in partial.items():
                grouped.setdefault(session_id, []).extend(chunks)
        for chunks in grouped.values():
            chunks.sort(key=lambda chunk: chunk.turn_index)
        return grouped

    def load_session_listing(self, session_ids: list[str]) -> dict[str, list[dict]]:
        """Load summary fields for a bounded set of sessions across shards."""
        grouped: dict[str, list[dict]] = {session_id: [] for session_id in session_ids}
        for _, db in self._iter_data_shards():
            partial = db.load_session_listing(session_ids)
            for session_id, rows in partial.items():
                grouped.setdefault(session_id, []).extend(rows)
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
        if self._data_dbs:
            return sum(db.chunk_count() for db in self._data_dbs)
        return self._index.chunk_count()

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
        if self._data_dbs:
            result: dict[int, str] = {}
            for shard_idx, db in self._iter_data_shards():
                for rowid, session_id in db.chunk_session_map().items():
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = session_id
            return result
        return self._index.chunk_session_map()

    def chunk_id_map(self) -> dict[int, str]:
        """Return a global ``{rowid: chunk_id}`` mapping for all chunks."""
        if self._data_dbs:
            result: dict[int, str] = {}
            for shard_idx, db in self._iter_data_shards():
                for rowid, chunk_id in db.chunk_id_map().items():
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = chunk_id
            return result
        return self._index.chunk_id_map()

    def get_chunk_id_rowid_map(self) -> dict[str, int]:
        """Return ``{chunk_id: global_rowid}`` across all shards."""
        if self._data_dbs:
            result: dict[str, int] = {}
            for shard_idx, db in self._iter_data_shards():
                for chunk_id, rowid in db.get_chunk_id_rowid_map().items():
                    result[chunk_id] = self._encode_chunk_rowid(shard_idx, rowid)
            return result
        return self._index.get_chunk_id_rowid_map()

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

    def fts_search(self, query: str, limit: int = 100, **kwargs) -> list[tuple]:
        """FTS search across all shards, merging results by score.
        """
        if self._data_dbs:
            all_results = []
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search(query, limit=limit, **kwargs)
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                )
            # Sort by score descending (score is element [1])
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        return self._index.fts_search(query, limit=limit, **kwargs)

    def fts_search_raw(self, fts_query: str, limit: int = 100) -> list[tuple[int, float]]:
        """Execute a pre-built FTS query across all shards."""
        if self._data_dbs:
            all_results = []
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search_raw(fts_query, limit=limit)
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                )
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        return self._index.fts_search_raw(fts_query, limit=limit)

    def fts_search_by_session(
        self,
        query: str,
        session_ids: list[str],
        limit: int = 100,
    ) -> list[tuple[int, float]]:
        """Session-scoped FTS search across all data shards."""
        if self._data_dbs:
            all_results = []
            for shard_idx, db in self._iter_data_shards():
                shard_hits = db.fts_search_by_session(query, session_ids, limit=limit)
                all_results.extend(
                    (self._encode_chunk_rowid(shard_idx, rowid), score)
                    for rowid, score in shard_hits
                )
            all_results.sort(key=lambda r: r[1], reverse=True)
            return all_results[:limit]
        return self._index.fts_search_by_session(query, session_ids, limit=limit)

    def get_embeddings(self, rowids: list[int]) -> dict[int, list[float]]:
        """Fetch chunk embeddings keyed by global sharded rowids."""
        if self._data_dbs:
            result: dict[int, list[float]] = {}
            for shard_idx, local_rowids in self._group_encoded_rowids(rowids).items():
                db = self._data_dbs[shard_idx - 1]
                for rowid, emb in db.get_embeddings(local_rowids).items():
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = emb
            return result
        return self._index.get_embeddings(rowids)

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
        if self._data_dbs:
            result: dict[int, list[float]] = {}
            for shard_idx, db in self._iter_data_shards():
                for rowid, emb in db.get_all_embeddings().items():
                    result[self._encode_chunk_rowid(shard_idx, rowid)] = emb
            return result
        return self._index.get_all_embeddings()

    def get_all_embeddings_numpy(self) -> "tuple[np.ndarray, list[int]]":
        """Load chunk embeddings into a numpy matrix with global rowids."""
        import numpy as np
        if self._data_dbs:
            matrices, all_rowids = [], []
            for shard_idx, db in self._iter_data_shards():
                mat, rids = db.get_all_embeddings_numpy()
                if mat.shape[0] > 0:
                    matrices.append(mat)
                    all_rowids.extend(
                        self._encode_chunk_rowid(shard_idx, r) for r in rids
                    )
            if not matrices:
                return np.empty((0, 384), dtype=np.float32), []
            return np.vstack(matrices), all_rowids
        return self._index.get_all_embeddings_numpy()

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
