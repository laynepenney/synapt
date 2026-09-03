from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from crewai.memory import Memory
    from crewai.memory.storage.backend import MemoryRecord, ScopeInfo
except ImportError as exc:  # pragma: no cover
    Memory = MemoryRecord = ScopeInfo = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_crewai() -> None:
    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError("Install synapt[crewai] to use the CrewAI adapter") from _IMPORT_ERROR


def _storage_path(project_dir: str | Path | None, path: str | Path | None) -> Path:
    if path:
        return Path(path).resolve()
    from synapt.recall.core import project_data_dir

    root = project_data_dir(Path(project_dir) if project_dir else None) / "crewai-memory.jsonl"
    return root.resolve()


def _recall_save(**kwargs: Any) -> str:
    from synapt.recall.server import recall_save

    return recall_save(**kwargs)


class SynaptStorage:
    """CrewAI storage backend backed by local recall persistence."""

    def __init__(self, project_dir: str | Path | None = None, path: str | Path | None = None) -> None:
        _require_crewai()
        self.path = _storage_path(project_dir, path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        return [MemoryRecord.model_validate(json.loads(line)) for line in self.path.read_text().splitlines() if line.strip()]

    def _write(self, records: list[MemoryRecord]) -> None:
        text = "\n".join(json.dumps(r.model_dump(mode="json")) for r in records)
        self.path.write_text(f"{text}\n" if text else "")

    def _match(self, record: MemoryRecord, scope_prefix: str | None, categories: list[str] | None, metadata_filter: dict[str, Any] | None) -> bool:
        if scope_prefix and not record.scope.startswith(scope_prefix):
            return False
        if categories and not set(categories).issubset(set(record.categories)):
            return False
        if metadata_filter and any(record.metadata.get(k) != v for k, v in metadata_filter.items()):
            return False
        return True

    def _score(self, left: list[float] | None, right: list[float]) -> float:
        if not left:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
        return dot / norm if norm else 0.0

    def save(self, records: list[MemoryRecord]) -> None:
        from synapt.recall.knowledge import VALID_CATEGORIES

        saved = {r.id: r for r in self._load()}
        for record in records:
            saved[record.id] = record
            # A CrewAI record's own categories are that framework's
            # vocabulary, not synapt's -- recall_save now refuses an
            # unrecognized category (a sibling fix) rather than silently
            # coercing it, so validate here and fall back exactly as the
            # coercion always did, rather than letting an unmatched
            # CrewAI category value fail the save outright.
            crewai_category = record.categories[0] if record.categories else None
            category = crewai_category if crewai_category in VALID_CATEGORIES else "workflow"
            _recall_save(
                content=record.content,
                category=category,
                confidence=record.importance,
                tags=record.categories or None,
                node_id=record.id,
            )
        self._write(sorted(saved.values(), key=lambda r: r.created_at))

    def search(self, query_embedding: list[float], scope_prefix: str | None = None, categories: list[str] | None = None, metadata_filter: dict[str, Any] | None = None, limit: int = 10, min_score: float = 0.0) -> list[tuple[MemoryRecord, float]]:
        matches = []
        for record in self._load():
            if not self._match(record, scope_prefix, categories, metadata_filter):
                continue
            score = self._score(record.embedding, query_embedding)
            if score >= min_score:
                matches.append((record, score))
        return sorted(matches, key=lambda item: item[1], reverse=True)[:limit]

    def delete(self, scope_prefix: str | None = None, categories: list[str] | None = None, record_ids: list[str] | None = None, older_than=None, metadata_filter: dict[str, Any] | None = None) -> int:
        keep, removed = [], 0
        for record in self._load():
            doomed = (record_ids and record.id in record_ids) or (older_than and record.created_at < older_than) or (not record_ids and self._match(record, scope_prefix, categories, metadata_filter))
            if doomed:
                removed += 1
                _recall_save(node_id=record.id, retract=True)
            else:
                keep.append(record)
        self._write(keep)
        return removed

    def update(self, record: MemoryRecord) -> None:
        self.save([record])

    def get_record(self, record_id: str) -> MemoryRecord | None:
        return next((r for r in self._load() if r.id == record_id), None)

    def list_records(self, scope_prefix: str | None = None, limit: int = 200, offset: int = 0) -> list[MemoryRecord]:
        records = [r for r in self._load() if self._match(r, scope_prefix, None, None)]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset : offset + limit]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        records = [r for r in self._load() if r.scope.startswith(scope)]
        depth = 0 if scope == "/" else len([part for part in scope.strip("/").split("/") if part])
        child_scopes = sorted(
            {
                "/" + "/".join(parts[: depth + 1])
                for r in records
                for parts in [[part for part in r.scope.strip("/").split("/") if part]]
                if len(parts) > depth
            }
        )
        return ScopeInfo(path=scope, record_count=len(records), categories=sorted({c for r in records for c in r.categories}), oldest_record=min((r.created_at for r in records), default=None), newest_record=max((r.created_at for r in records), default=None), child_scopes=child_scopes)

    def list_scopes(self, parent: str = "/") -> list[str]:
        return self.get_scope_info(parent).child_scopes

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.list_records(scope_prefix=scope_prefix):
            for category in record.categories:
                counts[category] = counts.get(category, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        return len(self.list_records(scope_prefix=scope_prefix))

    def reset(self, scope_prefix: str | None = None) -> None:
        self.delete(scope_prefix=scope_prefix)

    async def asave(self, records: list[MemoryRecord]) -> None:
        self.save(records)

    async def asearch(self, query_embedding: list[float], scope_prefix: str | None = None, categories: list[str] | None = None, metadata_filter: dict[str, Any] | None = None, limit: int = 10, min_score: float = 0.0) -> list[tuple[MemoryRecord, float]]:
        return self.search(query_embedding, scope_prefix, categories, metadata_filter, limit, min_score)

    async def adelete(self, scope_prefix: str | None = None, categories: list[str] | None = None, record_ids: list[str] | None = None, older_than=None, metadata_filter: dict[str, Any] | None = None) -> int:
        return self.delete(scope_prefix, categories, record_ids, older_than, metadata_filter)


def SynaptMemory(*, project_dir: str | Path | None = None, path: str | Path | None = None, **kwargs: Any) -> Memory:
    """Build a CrewAI Memory instance backed by synapt."""
    _require_crewai()
    return Memory(storage=SynaptStorage(project_dir=project_dir, path=path), **kwargs)
