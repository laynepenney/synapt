"""TDD spec: OSS inference tap at _make_recall_infer.

Every current extract-path model call flows through ``_make_recall_infer``'s
``recall_infer`` closure: B1 extraction, B2 decisions, B3 conflict judging,
and B4 composition. One ``infer`` UsageEvent is emitted per actual call,
stage-tagged in ``detail``, with real ``tokens_in``/``tokens_out`` from the
client's additive ``usage_out`` parameter.

``cached_tokens`` is structurally ``None`` here. The tap does not publish a
cache count, even if a client supplies one. A discriminating test below pins
that normalization rather than merely passing through a null fixture value.

Mutation gate: sever the emit_usage_event call inside recall_infer -> RED.
"""

from __future__ import annotations

import pytest

pytest.importorskip("synapt.recall.usage", reason="TDD contract: seam not implemented yet")

import synapt.recall.consolidate as consolidate
from synapt.recall.usage import UsageEvent, clear_usage_sinks, register_usage_sink


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


class _RaisingClient:
    def chat(self, model, messages, temperature=0.1, usage_out=None, **kwargs):
        if usage_out is not None:
            usage_out["tokens_in"] = 13
            usage_out["tokens_out"] = None
            usage_out["cached_tokens"] = None
        raise RuntimeError("controlled backend failure")


class _ClientReportingCachedTokens(_FakeClient):
    def chat(self, model, messages, temperature=0.1, usage_out=None, **kwargs):
        result = super().chat(model, messages, temperature, usage_out, **kwargs)
        if usage_out is not None:
            usage_out["cached_tokens"] = 17
        return result


@pytest.mark.parametrize("stage", ("B1", "B2", "B3-conflict", "B4"))
def test_infer_call_emits_exactly_one_usage_event_with_real_token_counts(stage):
    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    infer = consolidate._make_recall_infer(_FakeClient(), "qwen3.5-4b")
    result = infer({"prompt": "extract this", "messages": [], "stage": stage})

    assert result == "the completion text"
    assert len(received) == 1
    event = received[0]
    assert event.op == "infer"
    assert event.detail == stage
    assert event.model == "qwen3.5-4b"
    assert event.tokens_in == 88
    assert event.tokens_out == 41
    assert event.duration_ms is not None
    assert event.duration_ms >= 0


def test_failed_infer_still_emits_the_actual_call_and_preserves_the_error():
    """A failed retry is still consumed work and must remain visible to the meter."""
    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    infer = consolidate._make_recall_infer(_RaisingClient(), "qwen3.5-4b")
    with pytest.raises(RuntimeError, match="controlled backend failure"):
        infer({"prompt": "retry this", "messages": [], "stage": "B4-retry"})

    assert len(received) == 1
    event = received[0]
    assert event.detail == "B4-retry"
    assert event.tokens_in == 13
    assert event.tokens_out is None
    assert event.duration_ms is not None


def test_cached_tokens_is_structurally_none_for_the_self_hosted_tap():
    """The tap publishes no cache count even if its client supplies one."""
    received: list[UsageEvent] = []
    register_usage_sink(received.append)

    infer = consolidate._make_recall_infer(_ClientReportingCachedTokens(), "qwen3.5-4b")
    infer({"prompt": "extract this", "messages": [], "stage": "B4"})

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


def test_b3_conflict_judge_marks_its_infer_request_before_calling_the_seam():
    requests = []

    def infer(request):
        requests.append(request)
        return "COMPATIBLE"

    judge = consolidate._local_conflict_judge(infer)
    assert judge("candidate", "existing") is False
    assert requests[0]["stage"] == "B3-conflict"
