#!/usr/bin/env python3
"""Recovery for a store that missed knowledge writes while routing was
broken (a sharded store whose knowledge sync landed in the wrong file
before the routing fix): resync the live store from ``knowledge.jsonl``,
the actual source of truth for knowledge content.

``recall.db`` (the file the routing bug wrote into) is NOT a valid
recovery source: measured on a real store, it had accumulated years of
its own drift independent of ``knowledge.jsonl`` -- stale test fixtures,
long-superseded entries, contradicted nodes that were never pruned from
it. A diff against ``recall.db`` would copy hundreds of dead rows back
into production alongside the handful of genuinely missing ones. This
script never opens ``recall.db`` at all; it only ever reads
``knowledge.jsonl`` and writes into whichever file the on-disk layout
says is live.

Idempotent: re-running after a successful apply finds zero missing
nodes and does nothing. Dry-run by default; --apply is required to
write. This is exactly what a normal consolidation run's own
``_sync_knowledge_to_db`` already does the next time it runs; this
script exists to do it NOW, once, without forcing a full consolidation
pass (LLM calls, extraction, everything else consolidation triggers).

Usage:
    python scripts/resync-knowledge-from-jsonl.py <project_dir>            # dry run
    python scripts/resync-knowledge-from-jsonl.py <project_dir> --apply    # writes

<project_dir> is the recall project root (the directory whose
``.synapt/recall/`` holds ``knowledge.jsonl`` and the index), not the
index directory itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synapt.recall.core import project_data_dir, project_index_dir  # noqa: E402
from synapt.recall.knowledge import read_nodes  # noqa: E402
from synapt.recall.sharding import live_store_path  # noqa: E402
from synapt.recall.storage import RecallDB  # noqa: E402


def find_missing_nodes(project_dir: Path) -> list[dict]:
    """Nodes present in knowledge.jsonl, absent from the live store, by id.

    Read-only. Raises FileNotFoundError if knowledge.jsonl or the live
    store don't exist. Never opens recall.db.
    """
    kn_path = project_data_dir(project_dir) / "knowledge.jsonl"
    if not kn_path.exists():
        raise FileNotFoundError(f"no knowledge.jsonl at {kn_path} -- nothing to resync from")

    index_dir = project_index_dir(project_dir)
    live_path = live_store_path(index_dir)
    if not live_path.exists():
        raise FileNotFoundError(f"no live store at {live_path} -- nothing to resync into")

    jsonl_nodes = {n.id: n.to_dict() for n in read_nodes(kn_path)}

    live_db = RecallDB.open_readonly(live_path)
    try:
        live_nodes = {n["id"] for n in live_db.load_knowledge_nodes(status=None)}
    finally:
        live_db.close()

    return [node for node_id, node in jsonl_nodes.items() if node_id not in live_nodes]


def apply_resync(project_dir: Path, missing: list[dict]) -> None:
    """Write the missing nodes into the live store. Never opens recall.db."""
    index_dir = project_index_dir(project_dir)
    live_path = live_store_path(index_dir)
    db = RecallDB(live_path)
    try:
        db.save_knowledge_nodes(missing)
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", type=Path, help="the recall project root")
    parser.add_argument("--apply", action="store_true", help="write the missing nodes (default: dry run)")
    args = parser.parse_args()

    project_dir: Path = args.project_dir.resolve()

    try:
        missing = find_missing_nodes(project_dir)
    except FileNotFoundError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"project: {project_dir}")
    print(f"nodes in knowledge.jsonl, absent from the live store: {len(missing)}")
    for node in sorted(missing, key=lambda n: n.get("created_at", "")):
        preview = (node.get("content") or "")[:80]
        print(f"  {node['id']}  created={node.get('created_at', '?')}  {preview!r}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to write these nodes into the live store.")
        return 0

    if not missing:
        print("\nNothing to apply — already reconciled.")
        return 0

    apply_resync(project_dir, missing)

    remaining = find_missing_nodes(project_dir)
    print(f"\nAPPLIED. Re-checked: {len(remaining)} nodes still missing (expect 0).")
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
