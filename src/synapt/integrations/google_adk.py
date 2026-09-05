"""Google ADK memory service backed by synapt recall.

Provides SynaptMemoryService, a BaseMemoryService implementation that
stores conversation events in recall's knowledge graph and uses hybrid
search for retrieval. Drop-in replacement for InMemoryMemoryService.

Usage:
    from google.adk.agents import LlmAgent
    from synapt.integrations.google_adk import SynaptMemoryService

    memory = SynaptMemoryService()

    agent = LlmAgent(
        model="gemini-2.5-flash",
        name="my_agent",
        memory_service=memory,
    )
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from synapt.recall.core import project_data_dir, project_index_dir
from synapt.recall.sharding import live_store_path
from synapt.recall.storage import RecallDB

try:
    from google.adk.memory.base_memory_service import (
        BaseMemoryService,
        SearchMemoryResponse,
    )
    from google.adk.memory.memory_entry import MemoryEntry
    from google.genai import types
except ImportError as exc:
    BaseMemoryService = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session

log = logging.getLogger("synapt.integrations.google_adk")


def _require_adk() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "Install google-adk to use the Google ADK memory backend"
        ) from _IMPORT_ERROR


def _event_text(event: Event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return " ".join(
        part.text for part in event.content.parts if getattr(part, "text", None)
    )


def _session_key(app_name: str, user_id: str) -> str:
    return f"{app_name}/{user_id}"


def _node_id(app_name: str, user_id: str, content_hash: str) -> str:
    raw = f"{app_name}/{user_id}/{content_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _words(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[A-Za-z0-9_]+", text)}


if BaseMemoryService is not None:

    class SynaptMemoryService(BaseMemoryService):
        """Google ADK MemoryService backed by synapt recall.

        Replaces the default InMemoryMemoryService with recall-powered
        hybrid search. Session events are persisted as recall knowledge
        nodes; search_memory uses semantic + keyword hybrid retrieval.

        Args:
            max_search_results: Maximum number of memory entries to return
                from search_memory (default: 10).
            project_root: Store scope for both writes and reads. Threaded
                through to project_data_dir/project_index_dir instead of
                their ambient (cwd/env) resolution, so a caller (a test, a
                sandboxed agent run) can point this service at an isolated
                store rather than the ambient real one. None keeps the
                existing ambient-resolution behavior.
        """

        def __init__(
            self,
            *,
            max_search_results: int = 10,
            project_root: str | Path | None = None,
        ) -> None:
            _require_adk()
            self._max_results = max_search_results
            self._project_root = Path(project_root) if project_root is not None else None
            self._lock = threading.Lock()
            self._ingested: dict[str, set[str]] = {}

        async def add_session_to_memory(self, session: Session) -> None:
            key = _session_key(session.app_name, session.user_id)

            with self._lock:
                if key not in self._ingested:
                    self._ingested[key] = set()
                seen = self._ingested[key]

            for event in session.events:
                text = _event_text(event)
                if not text or event.id in seen:
                    continue
                self._save_to_recall(
                    app_name=session.app_name,
                    user_id=session.user_id,
                    text=text,
                    author=getattr(event, "author", "unknown"),
                )
                with self._lock:
                    seen.add(event.id)

        async def add_events_to_memory(
            self,
            *,
            app_name: str,
            user_id: str,
            events: Sequence[Event],
            session_id: str | None = None,
            custom_metadata: Mapping[str, object] | None = None,
        ) -> None:
            key = _session_key(app_name, user_id)

            with self._lock:
                if key not in self._ingested:
                    self._ingested[key] = set()
                seen = self._ingested[key]

            for event in events:
                text = _event_text(event)
                if not text or event.id in seen:
                    continue
                self._save_to_recall(
                    app_name=app_name,
                    user_id=user_id,
                    text=text,
                    author=getattr(event, "author", "unknown"),
                )
                with self._lock:
                    seen.add(event.id)

        async def add_memory(
            self,
            *,
            app_name: str,
            user_id: str,
            memories: Sequence[MemoryEntry],
            custom_metadata: Mapping[str, object] | None = None,
        ) -> None:
            for entry in memories:
                text = ""
                if entry.content and entry.content.parts:
                    text = " ".join(
                        part.text
                        for part in entry.content.parts
                        if getattr(part, "text", None)
                    )
                if not text:
                    continue
                self._save_to_recall(
                    app_name=app_name,
                    user_id=user_id,
                    text=text,
                    author=entry.author or "unknown",
                )

        async def search_memory(
            self,
            *,
            app_name: str,
            user_id: str,
            query: str,
        ) -> SearchMemoryResponse:
            try:
                entries = self._search_scoped_recall(
                    app_name=app_name,
                    user_id=user_id,
                    query=query,
                )
            except Exception as e:
                log.warning("Recall search failed: %s", e)
                return SearchMemoryResponse()

            if not entries:
                return SearchMemoryResponse()

            return SearchMemoryResponse(memories=entries[:self._max_results])

        def _save_to_recall(
            self,
            *,
            app_name: str,
            user_id: str,
            text: str,
            author: str,
        ) -> None:
            try:
                # recall_save() (synapt.recall.server) hard-codes its own
                # project scope to None -- ambient resolution via
                # SYNAPT_RECALL_ROOT / GRIPSPACE_ROOT, deliberately, so it
                # can't be handed self._project_root without either mutating
                # process-wide env vars around this call (unsafe: this class
                # is used from concurrent async callers sharing one process,
                # per the threading.Lock above) or reaching past the wrapper
                # to the same primitives it uses. This calls those primitives
                # directly, parameterized on self._project_root, mirroring
                # recall_save's create path (append the durable JSONL record,
                # upsert the queryable SQLite row _search_scoped_recall
                # reads) minus the parts this integration doesn't need yet
                # (retract, existing-version bump, embeddings).
                from synapt.recall.knowledge import KnowledgeNode, append_node

                content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
                node = KnowledgeNode.create(
                    content=text,
                    # "conversation" is not one of synapt's VALID_CATEGORIES --
                    # recall_save refuses an unrecognized category rather
                    # than silently coercing it, so this always used
                    # "workflow" under the hood anyway. Pass it explicitly.
                    category="workflow",
                    confidence=0.8,
                    tags=[
                        "google-adk",
                        f"app:{app_name}",
                        f"user:{user_id}",
                        f"author:{author}",
                    ],
                    node_id=_node_id(app_name, user_id, content_hash),
                )
                append_node(
                    node, project_data_dir(self._project_root) / "knowledge.jsonl"
                )
                db = RecallDB(live_store_path(project_index_dir(self._project_root)))
                try:
                    db.upsert_knowledge_node(node.to_dict())
                finally:
                    db.close()
            except Exception as e:
                log.warning("Recall save failed: %s", e)

        def _search_scoped_recall(
            self,
            *,
            app_name: str,
            user_id: str,
            query: str,
        ) -> list[MemoryEntry]:
            query_words = _words(query)
            if not query_words:
                return []

            app_tag = f"app:{app_name}"
            user_tag = f"user:{user_id}"
            db = RecallDB(live_store_path(project_index_dir(self._project_root)))
            try:
                nodes = db.load_knowledge_nodes(status="active")
            finally:
                db.close()

            scored: list[tuple[float, dict[str, Any]]] = []
            for node in nodes:
                tags = set(node.get("tags", []))
                if "google-adk" not in tags or app_tag not in tags or user_tag not in tags:
                    continue
                content = node.get("content", "")
                content_words = _words(content)
                if not content_words:
                    continue

                overlap = len(query_words & content_words)
                if overlap == 0:
                    continue

                score = float(overlap)
                if query.lower() in content.lower():
                    score += 0.5
                scored.append((score, node))

            scored.sort(key=lambda item: item[0], reverse=True)
            entries: list[MemoryEntry] = []
            for _, node in scored[: self._max_results]:
                tags = node.get("tags", [])
                author = "recall"
                for tag in tags:
                    if tag.startswith("author:"):
                        author = tag.split(":", 1)[1] or "recall"
                        break
                entries.append(
                    MemoryEntry(
                        content=types.Content(
                            parts=[types.Part(text=node.get("content", ""))],
                            role="model",
                        ),
                        author=author,
                        timestamp=node.get("updated_at") or node.get("created_at"),
                    )
                )
            return entries

else:

    class SynaptMemoryService:  # type: ignore[no-redef]
        """Stub when google-adk is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            _require_adk()
