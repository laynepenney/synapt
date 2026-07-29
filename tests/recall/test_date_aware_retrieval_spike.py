"""Date-aware retrieval tests for structured temporal fixtures."""

from datetime import datetime, timezone

import pytest

from synapt.recall.core import TranscriptChunk, TranscriptIndex


def _chunk(chunk_id: str, date: str, text: str) -> TranscriptChunk:
    return TranscriptChunk(
        id=chunk_id,
        session_id=chunk_id.replace(":", "-")[:36].ljust(36, "0"),
        timestamp=date,
        turn_index=0,
        user_text=text,
        assistant_text="",
    )


def _lookup_text(
    query: str,
    chunks: list[TranscriptChunk],
    now: datetime | None = None,
) -> str:
    index = TranscriptIndex(chunks)
    return index.lookup(
        query,
        max_chunks=10,
        max_tokens=4000,
        threshold_ratio=0,
        now=now,
    )


def test_lookup_uses_reference_clock_for_missing_year_temporal_queries():
    result = _lookup_text(
        "What did I pray about on May 31?",
        [
            _chunk(
                "Calvin:2023",
                "2023-05-31",
                "I prayed with gratitude after the Tokyo concert.",
            ),
            _chunk(
                "Calvin:2026",
                "2026-05-31",
                "I prayed about a future album release.",
            ),
        ],
        now=datetime(2023, 6, 2, tzinfo=timezone.utc),
    )

    assert "Tokyo concert" in result
    assert "future album release" not in result


TEMPORAL_FIXTURE_CASES = [
    (
        "temporal:Calvin:01",
        "What did I pray about on May 31, 2023?",
        [(
            "Calvin:conv-50:session_7:event_summary:Calvin:1:1",
            "2023-05-31",
            "expected Calvin May 31 Tokyo concert prayer",
        )],
    ),
    (
        "temporal:Caroline:02",
        "Can you remind me what I prayed for on May 8 and 9, 2023?",
        [
            (
                "Caroline:conv-26:session_1:event_summary:Caroline:1:1",
                "2023-05-08",
                "expected Caroline May 8 courage before entering the first community meeting",
            ),
            (
                "Caroline:conv-26:session_1:event_summary:Caroline:1:2",
                "2023-05-09",
                "expected Caroline May 9 clarity after a hard conversation with her brother",
            ),
        ],
    ),
    (
        "temporal:Evan:03",
        "What was my prayer on January 1, 2024?",
        [(
            "Evan:conv-49:session_22:event_summary:Evan:2:2",
            "2024-01-01",
            "expected Evan January 1 prayer",
        )],
    ),
    (
        "temporal:Jolene:04",
        "What did I pray for on March 28 and 29, 2023?",
        [
            (
                "Jolene:conv-48:session_11:event_summary:Jolene:1:1",
                "2023-03-28",
                "expected Jolene March 28 courage before the oncology appointment",
            ),
            (
                "Jolene:conv-48:session_11:event_summary:Jolene:1:2",
                "2023-03-29",
                "expected Jolene March 29 gratitude after the lab results brought relief",
            ),
        ],
    ),
    (
        "temporal:Jon:05",
        "What was my prayer on August 6, 2023?",
        [(
            "Jon:conv-30:session_19:observation:Jon:3:2",
            "2023-08-06",
            "expected Jon August 6 prayer",
        )],
    ),
    (
        "temporal:James:06",
        "Can you tell me what I prayed about on November 1, 2022?",
        [(
            "James:conv-47:session_29:event_summary:James:1:2",
            "2022-11-01",
            "expected James November 1 prayer",
        )],
    ),
    (
        "temporal:Joanna:07",
        "What did I pray for on April 17, 2022?",
        [(
            "Joanna:conv-42:session_8:observation:Joanna:3:1",
            "2022-04-17",
            "expected Joanna April 17 prayer",
        )],
    ),
    (
        "temporal:Andrew:08",
        "What was my prayer on October 1, 2023?",
        [(
            "Andrew:conv-44:session_20:observation:Andrew:6:1",
            "2023-10-01",
            "expected Andrew October 1 prayer",
        )],
    ),
]


@pytest.mark.parametrize("fixture_id,query,expected_chunks", TEMPORAL_FIXTURE_CASES)
def test_all_current_temporal_fixture_queries_filter_by_structured_date(
    fixture_id: str,
    query: str,
    expected_chunks: list[tuple[str, str, str]],
):
    chunks = [
        _chunk(chunk_id, date, text)
        for chunk_id, date, text in expected_chunks
    ]
    chunks.extend([
        _chunk(
            f"{fixture_id}:noise-before",
            "2022-01-01",
            f"noise before {fixture_id}",
        ),
        _chunk(
            f"{fixture_id}:noise-after",
            "2025-01-01",
            f"noise after {fixture_id}",
        ),
    ])

    result = _lookup_text(query, chunks)

    for _, _, text in expected_chunks:
        assert text in result
    assert f"noise before {fixture_id}" not in result
    assert f"noise after {fixture_id}" not in result


def test_exact_date_query_retrieves_calvin_may_31_fixture_shape():
    result = _lookup_text(
        "What did I pray about on May 31, 2023?",
        [
            _chunk(
                "Calvin:conv-50:session_7:event_summary:Calvin:1:1",
                "2023-05-31",
                "Thank you for the Frank Ocean Tour in Tokyo and the shared moments on stage.",
            ),
            _chunk(
                "Calvin:other",
                "2023-05-30",
                "Please help me prepare for tomorrow's performance.",
            ),
        ],
    )

    assert "Frank Ocean Tour in Tokyo" in result
    assert "tomorrow's performance" not in result


def test_exact_date_query_retrieves_james_november_1_fixture_shape():
    result = _lookup_text(
        "Can you tell me what I prayed about on November 1, 2022?",
        [
            _chunk(
                "James:conv-47:session_29:event_summary:James:1:2",
                "2022-11-01",
                "I asked for steadiness while navigating a fragile family conversation.",
            ),
            _chunk(
                "James:other",
                "2022-10-31",
                "I prayed about patience before the holiday gathering.",
            ),
        ],
    )

    assert "fragile family conversation" in result
    assert "holiday gathering" not in result


def test_paired_date_query_retrieves_caroline_may_8_and_9_fixture_shape():
    result = _lookup_text(
        "Can you remind me what I prayed for on May 8 and 9, 2023?",
        [
            _chunk(
                "Caroline:conv-26:session_1:event_summary:Caroline:1:1",
                "2023-05-08",
                "I prayed for courage before starting the LGBTQ support group.",
            ),
            _chunk(
                "Caroline:conv-26:session_1:event_summary:Caroline:1:2",
                "2023-05-09",
                "I prayed for clarity after the first support group conversation.",
            ),
            _chunk(
                "Caroline:other",
                "2023-05-10",
                "I prayed about unrelated school scheduling.",
            ),
        ],
    )

    assert "starting the LGBTQ support group" in result
    assert "first support group conversation" in result
    assert "unrelated school scheduling" not in result


def test_paired_date_query_retrieves_jolene_march_28_and_29_fixture_shape():
    result = _lookup_text(
        "What did I pray for on March 28 and 29, 2023?",
        [
            _chunk(
                "Jolene:conv-48:session_11:event_summary:Jolene:1:1",
                "2023-03-28",
                "I prayed for peace while preparing for the medical appointment.",
            ),
            _chunk(
                "Jolene:conv-48:session_11:event_summary:Jolene:1:2",
                "2023-03-29",
                "I prayed with gratitude after the appointment brought better news.",
            ),
            _chunk(
                "Jolene:other",
                "2023-03-30",
                "I prayed about a different work conflict.",
            ),
        ],
    )

    assert "preparing for the medical appointment" in result
    assert "appointment brought better news" in result
    assert "different work conflict" not in result
