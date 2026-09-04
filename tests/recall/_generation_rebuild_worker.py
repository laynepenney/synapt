"""Worker for test_generations.py's real cross-process demonstration.

Run as a subprocess (never imported): opens the sharded index at argv[1]
and does a genuine full rebuild via ShardedRecallDB.save_chunks(), which
goes through generations.rebuild_and_publish end to end. A separate,
concurrent process (the test itself) polls the same index_dir through
ShardedRecallDB.open() while this runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from synapt.recall.core import TranscriptChunk
from synapt.recall.sharded_db import ShardedRecallDB
import synapt.recall.sharding as sharding_mod


def main() -> None:
    index_dir = Path(sys.argv[1])
    n_chunks = int(sys.argv[2])
    threshold = int(sys.argv[3])

    sharding_mod.SHARD_CHUNK_THRESHOLD = threshold

    chunks = [
        TranscriptChunk(
            id=f"real-build:t{i}",
            session_id="real-build-session",
            timestamp=f"2026-09-03T{(i // 3600) % 24:02d}:{(i // 60) % 60:02d}:{i % 60:02d}Z",
            turn_index=i,
            user_text=f"real build question {i}: what does the quality curve do",
            assistant_text=f"real build answer {i}: the quality curve uses a Hermite spline",
        )
        for i in range(n_chunks)
    ]

    db = ShardedRecallDB.open(index_dir)
    db.save_chunks(chunks)
    print(f"shard_count={db.shard_count} chunk_count={db.chunk_count()}")
    db.close()


if __name__ == "__main__":
    main()
