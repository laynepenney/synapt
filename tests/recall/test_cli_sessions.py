from argparse import Namespace
from unittest import mock

from synapt.recall.cli import cmd_sessions
from synapt.recall.core import TranscriptChunk
from synapt.recall.sharded_db import ShardedRecallDB
from synapt.recall.storage import RecallDB


def test_sessions_accepts_a_sharded_only_index(tmp_path, capsys):
    RecallDB(tmp_path / "index.db").close()
    shard = RecallDB(tmp_path / "data_001.db")
    shard.save_chunks(
        [
            TranscriptChunk(
                id="session-a:t0",
                session_id="session-a",
                timestamp="2026-08-25T10:00:00Z",
                turn_index=0,
                user_text="bounded session browsing",
                assistant_text="working",
            )
        ]
    )
    shard.close()

    with (
        mock.patch(
            "synapt.recall.core.TranscriptIndex.load",
            side_effect=AssertionError("session browsing constructed the full index"),
        ),
        mock.patch.object(
            ShardedRecallDB,
            "load_session_chunks_many",
            side_effect=AssertionError("session listing loaded full chunk bodies"),
        ),
    ):
        cmd_sessions(
            Namespace(
                index=str(tmp_path),
                out=None,
                max_sessions=20,
                after=None,
                before=None,
            )
        )

    assert "session-" in capsys.readouterr().out
