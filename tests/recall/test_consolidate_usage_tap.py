"""TDD spec (premium#743, OSS seam half): tap point 1 — _make_recall_infer.

Every extract-path model call (B1 identify, B2 decisions, B4 compose) flows
through ``_make_recall_infer``'s ``recall_infer`` closure. One ``infer``
UsageEvent per call, stage-tagged in ``detail``, real ``tokens_in``/``tokens_out``
from the client's new ``usage_out`` param (see test_model_client_usage_out.py).

``cached_tokens`` is STRUCTURALLY None here — this tap is the self-hosted
MLX/vLLM enrichment/consolidation models, which have no cache-read/write
BILLING tier the way hosted Anthropic/OpenAI APIs do. That's a separate,
not-yet-tapped surface (see the metering design doc's tap-point 1 note) — this
spec does not claim cached_tokens is ever non-null from this tap, and a test
below pins that it never is, so a future reader can't re-conflate the surfaces.

Mutation gate: sever the emit_usage_event call inside recall_infer -> RED.
"""

from __future__ import annotations

import synapt.recall.consolidate as consolidate
from synapt.recall.usage import UsageEvent, clear_usage_sinks, register_usage_sink
import pytest


@pytest.fixture(autouse=True)
def _clean_usage_state():
    clear_usage_sinks()
    yield
    clear_usage_sinks()


class _FakeClient:
    """Mimics ModelClient.chat()'s new usage_out-populating contract."""

    def chat(self, model, messages, temperature=0.1, usage_out=None, **kwargs):
        if usage_out is not None:
            usage_out["tokens_in"] = 88
            usage_out["tokens_out"] = 41
            usage_out["cached_tokens"] = None
        return "the completion text"


def test_infer_call_emits_exactly_one_usage_event_with_real_token_counts():
    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    infer = consolidate._make_recall_infer(_FakeClient(), "qwen3.5-4b")
    result = infer({"prompt": "extract this", "messages": [], "stage": "B2"})

    assert result == "the completion text"
    assert len(received) == 1
    event = received[0]
    assert event.op == "infer"
    assert event.detail == "B2"
    assert event.model == "qwen3.5-4b"
    assert event.tokens_in == 88
    assert event.tokens_out == 41


def test_cached_tokens_is_structurally_none_for_the_self_hosted_tap():
    """Self-hosted MLX/vLLM compute has no cache-read/write billing tier — this
    is correct, not a gap. The hosted-API surface (where cache reads ARE
    load-bearing, per the 84-day corpus) is a separate, future tap."""
    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    infer = consolidate._make_recall_infer(_FakeClient(), "qwen3.5-4b")
    infer({"prompt": "extract this", "messages": [], "stage": "B4-compose"})

    assert received[0].cached_tokens is None


def test_stage_tag_defaults_to_b1_when_the_request_carries_no_explicit_stage():
    """B1's BatchInferRequest shape is out of recall's control (synapt-extract
    owns it) — the same fallback _make_recall_infer's max_tokens estimation
    already documents applies here too."""
    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    infer = consolidate._make_recall_infer(_FakeClient(), "qwen3.5-4b")
    infer({"prompt": "extract this", "messages": []})  # no "stage" key

    assert received[0].detail == "B1"


def test_infer_tap_never_raises_when_no_sink_is_registered():
    infer = consolidate._make_recall_infer(_FakeClient(), "qwen3.5-4b")
    result = infer({"prompt": "extract this", "messages": [], "stage": "B2"})
    assert result == "the completion text"  # must not raise, no sink required
