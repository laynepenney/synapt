"""B4 mechanism gate: rejoin create-bound atoms before reconcile.

These are deliberately implementation-ahead TDD tests. Every action-item fixture crosses the
real prefilter -> ``extract_batch`` -> ``_decide_actions`` path with faithful fake inference.
No hand-built reconcile dict stands in for B2 output. The fake seam controls only model text.

B4's v1 truth guard is prompt-level. These tests pin the anti-corruption instructions and feed
faithful compositions. They do not invent a deterministic entailment validator that the design
explicitly leaves to v2 and the source-clean dogfood axis.
"""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

import synapt.recall.consolidate as consolidate
from synapt.recall.journal import JournalEntry
from synapt.recall.knowledge import KnowledgeNode, read_nodes


@dataclass(frozen=True)
class _FactSpec:
    text: str
    category: str = "fact"
    tags: tuple[str, ...] = ()
    action: str = "create"
    existing_id: str | None = None
    temporal_role: str | None = None
    resolved: str | None = None
    resolved_end: str | None = None


@dataclass
class _PipelineFruit:
    cluster_id: str
    cluster: list[JournalEntry]
    envelopes: list
    action_items: list[dict]
    existing_nodes: list[KnowledgeNode]


def _journal_cluster(specs: list[_FactSpec]) -> list[JournalEntry]:
    return [
        JournalEntry(
            timestamp="2025-03-01T09:00:00Z",
            session_id="s-b4-real-fruit",
            done=[spec.text for spec in specs],
        )
    ]


def _extraction_completion(spec: _FactSpec) -> str:
    facts = []
    decisions = []
    if spec.category == "decision":
        decisions.append({"text": spec.text})
    else:
        facts.append({"text": spec.text, "category": spec.category})

    temporal_refs = []
    if spec.temporal_role:
        ref = {
            "raw": "date from source",
            "role": spec.temporal_role,
            "resolved": spec.resolved,
        }
        if spec.resolved_end is not None:
            ref["resolved_end"] = spec.resolved_end
        temporal_refs.append(ref)

    return json.dumps({
        "extracted_at": "2025-03-01T09:00:00Z",
        "facts": facts,
        "decisions": decisions,
        "temporal_refs": temporal_refs,
    })


def _real_pipeline(
    specs: list[_FactSpec],
    *,
    cluster_id: str = "b4-real-fruit",
    existing_nodes: list[KnowledgeNode] | None = None,
) -> _PipelineFruit:
    """Produce B4 input through the actual B1 and B2 code paths.

    The assertions here are fruit pins. If extract's real envelope shape or B2's real ordering
    changes, the fixture fails instead of quietly manufacturing the old shape in a proxy map.
    """
    cluster = _journal_cluster(specs)
    existing = list(existing_nodes or [])

    def extract_infer(request):
        matches = [spec for spec in specs if spec.text in request["prompt"]]
        assert len(matches) == 1, "each real BatchUnit prompt must bind one source item"
        return _extraction_completion(matches[0])

    envelopes = consolidate._run_coro_blocking(
        consolidate._extract_cluster_units(cluster, cluster_id, extract_infer)
    )
    assert len(envelopes) == len(specs)
    assert all(envelope.status == "ok" for envelope in envelopes)
    assert [envelope.source_unit_id for envelope in envelopes] == [
        f"{cluster_id}:0:done:{index}" for index in range(len(specs))
    ]

    def action_infer(request):
        assert all(spec.text in request["prompt"] for spec in specs)
        return json.dumps({
            "actions": [
                {
                    "index": index,
                    "action": spec.action,
                    "existing_id": spec.existing_id,
                    "contradiction_note": (
                        "the newer source reverses the persisted claim"
                        if spec.action == "contradict" else ""
                    ),
                    "tags": list(spec.tags),
                }
                for index, spec in enumerate(specs)
            ]
        })

    action_items = consolidate._decide_actions(
        cluster, cluster_id, envelopes, existing, action_infer,
    )
    assert len(action_items) == len(specs)
    assert [item["content"] for item in action_items] == [spec.text for spec in specs]
    assert [item["action"] for item in action_items] == [spec.action for spec in specs]
    return _PipelineFruit(cluster_id, cluster, envelopes, action_items, existing)


def _invoke_rejoin(
    fruit: _PipelineFruit,
    completion,
    *,
    action_items: list[dict] | None = None,
    decision_log_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    requests: list[dict] = []

    def infer(request):
        requests.append(request)
        if isinstance(completion, BaseException):
            raise completion
        value = completion(request) if callable(completion) else completion
        return value if isinstance(value, str) else json.dumps(value)

    rejoin = getattr(consolidate, "_rejoin_create_actions")
    output = rejoin(
        deepcopy(action_items if action_items is not None else fruit.action_items),
        fruit.cluster_id,
        infer,
        decision_log_path=decision_log_path,
    )
    return output, requests


def _response(*groups: tuple[list[int], str]) -> dict:
    return {
        "groups": [
            {"indices": indices, "content": content}
            for indices, content in groups
        ]
    }


def _by_content(items: list[dict], content: str) -> dict:
    return next(item for item in items if item["content"] == content)


def _warning_messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]


@pytest.fixture(scope="module")
def truth_fruit() -> _PipelineFruit:
    return _real_pipeline([
        _FactSpec(
            "The adapter failures are unrelated to SQLite locking.",
            category="debugging",
            tags=("adapter", "locking"),
        ),
        _FactSpec(
            "PR #867 closes recall#865 by adding the missing Windows coverage.",
            category="decision",
            tags=("pr-867", "windows"),
        ),
        _FactSpec(
            "The report labels these outcomes as LOW cases, where LOW is a severity convention.",
            category="fact",
            tags=("quality", "severity"),
        ),
        _FactSpec(
            "Recall stores durable node revisions in an append-versioned knowledge.jsonl file.",
            category="architecture",
            tags=("recall", "storage"),
        ),
    ])


# Guard 1: count invariant, create-only scope, and never-drop behavior.


def test_b2_real_output_carries_each_source_unit_id(truth_fruit):
    """B4 provenance begins in B2, so bind the key on actual B2 output, not a proxy map."""
    assert [item["source_unit_id"] for item in truth_fruit.action_items] == [
        envelope.source_unit_id for envelope in truth_fruit.envelopes
    ]


def test_unaddressed_creates_pass_through_and_invented_indices_are_ignored(truth_fruit):
    composed = (
        "The adapter failures are unrelated to SQLite locking, and PR #867 closes "
        "recall#865 by adding the missing Windows coverage."
    )
    output, _ = _invoke_rejoin(
        truth_fruit,
        _response(
            ([0, 1], composed),
            ([999], "This invented index must never create a node."),
        ),
    )

    assert len(output) == 3
    assert {item["content"] for item in output} == {
        composed,
        truth_fruit.action_items[2]["content"],
        truth_fruit.action_items[3]["content"],
    }


def test_group_mixing_real_and_invented_indices_drops_only_that_group(truth_fruit):
    """A composition describing an invented member is unsafe even when some indices are real.

    Filtering ``999`` and trusting the same composed text for members 0 and 1 would persist text
    written for a different membership set. Drop that proposed group, pass its real members through
    unchanged, and still accept an independent valid group from the same one-pass response.
    """
    valid_composed = (
        "The report uses LOW as a severity convention, and Recall stores append-versioned "
        "knowledge.jsonl revisions."
    )
    output, _ = _invoke_rejoin(
        truth_fruit,
        _response(
            ([0, 1, 999], "This composition was written for a fabricated membership set."),
            ([2, 3], valid_composed),
        ),
    )

    assert len(output) == 3
    assert truth_fruit.action_items[0] in output
    assert truth_fruit.action_items[1] in output
    assert _by_content(output, valid_composed)["content"] == valid_composed
    assert all(
        item["content"] != "This composition was written for a fabricated membership set."
        for item in output
    )


def test_singleton_group_passes_the_original_item_through_unchanged(truth_fruit):
    output, _ = _invoke_rejoin(
        truth_fruit,
        _response(
            ([0], "The model must not rewrite a singleton."),
            ([1, 2, 3], "PR #867 closes recall#865; LOW remains the severity convention; "
             "Recall stores append-versioned knowledge.jsonl revisions."),
        ),
    )

    assert truth_fruit.action_items[0] in output
    assert all(item["content"] != "The model must not rewrite a singleton." for item in output)


def test_duplicate_membership_fails_open_the_whole_cluster_with_marker(truth_fruit, caplog):
    caplog.set_level(logging.WARNING)
    output, _ = _invoke_rejoin(
        truth_fruit,
        _response(
            ([0, 1], "first invalid composition"),
            ([1, 2], "second invalid composition"),
        ),
    )

    assert output == truth_fruit.action_items
    messages = _warning_messages(caplog)
    assert any("B4_COMPOSE_FAIL_OPEN" in message for message in messages)
    assert any(truth_fruit.cluster_id in message for message in messages)
    assert any("duplicate" in message.lower() for message in messages)


def test_only_create_actions_enter_b4_and_other_actions_pass_through_untouched():
    corroborated = KnowledgeNode.create(
        content="The persisted adapter contract already names the production retry boundary.",
        category="architecture",
        node_id="kn-b4-existing-a",
    )
    contradicted = KnowledgeNode.create(
        content="The persisted deployment record says the Windows job is intentionally disabled.",
        category="decision",
        node_id="kn-b4-existing-b",
    )
    specs = [
        _FactSpec("The sprint-41 branch keeps the B4 stage behind SYNAPT_USE_EXTRACT.", tags=("b4",)),
        _FactSpec(
            "The production retry boundary matches the persisted adapter contract.",
            action="corroborate",
            existing_id=corroborated.id,
        ),
        _FactSpec("The B4 response carries indexed member groups into deterministic guards.", tags=("b4",)),
        _FactSpec(
            "The Windows job is now required rather than intentionally disabled.",
            action="contradict",
            existing_id=contradicted.id,
        ),
    ]
    fruit = _real_pipeline(specs, existing_nodes=[corroborated, contradicted])
    composed = (
        "The sprint-41 B4 stage stays behind SYNAPT_USE_EXTRACT and guards indexed member "
        "groups deterministically."
    )
    output, requests = _invoke_rejoin(fruit, _response(([0, 1], composed)))

    prompt = requests[0]["prompt"]
    assert specs[0].text in prompt and specs[2].text in prompt
    assert specs[1].text not in prompt and specs[3].text not in prompt
    assert [item for item in output if item["action"] != "create"] == [
        fruit.action_items[1], fruit.action_items[3],
    ]
    assert _by_content(output, composed)["action"] == "create"


def test_no_create_actions_skip_inference_and_pass_through():
    existing = KnowledgeNode.create(
        content="The persisted deployment contract requires the Windows validation matrix.",
        category="decision",
        node_id="kn-b4-existing-only",
    )
    fruit = _real_pipeline(
        [
            _FactSpec(
                "The current deployment contract still requires the Windows validation matrix.",
                action="corroborate",
                existing_id=existing.id,
            ),
            _FactSpec(
                "The new release record reverses the persisted deployment contract.",
                action="contradict",
                existing_id=existing.id,
            ),
        ],
        existing_nodes=[existing],
    )

    output, requests = _invoke_rejoin(
        fruit,
        AssertionError("B4 must not infer when there are no create actions"),
    )
    assert output == fruit.action_items
    assert requests == []


# Guard 2: model-visible anti-corruption contract. Runtime entailment remains v2.


def test_composition_prompt_pins_the_three_observed_truth_corruption_classes(truth_fruit):
    faithful = (
        "The adapter failures are unrelated to SQLite locking. PR #867 closes recall#865. "
        "The report calls the outcomes LOW cases under its severity convention."
    )
    output, requests = _invoke_rejoin(
        truth_fruit,
        _response(([0, 1, 2], faithful), ([3], truth_fruit.action_items[3]["content"])),
    )

    assert len(requests) == 1
    prompt = requests[0]["prompt"].lower()
    assert "member" in prompt and "only" in prompt
    assert "negation" in prompt
    assert "convention" in prompt
    assert "relation" in prompt
    assert "caus" in prompt
    assert "qualifier" in prompt
    assert _by_content(output, faithful)["content"] == faithful


def test_faithful_composed_text_is_not_rewritten_after_the_model_response(truth_fruit):
    faithful = (
        "PR #867 closes recall#865, while the adapter failures remain unrelated to SQLite "
        "locking and LOW remains a severity label."
    )
    output, _ = _invoke_rejoin(
        truth_fruit,
        _response(([0, 1, 2], faithful), ([3], truth_fruit.action_items[3]["content"])),
    )
    assert _by_content(output, faithful)["content"] == faithful


# Guard 3: deterministic metadata and durable decision-log provenance.


def test_metadata_unions_tags_and_uses_member_category_majority():
    fruit = _real_pipeline([
        _FactSpec("The B4 adapter groups the first indexed release fact.", "architecture", ("b4", "shared")),
        _FactSpec("The B4 adapter groups the second indexed release fact.", "architecture", ("second", "shared")),
        _FactSpec("The release record also captures the operator decision.", "decision", ("operator",)),
    ])
    composed = (
        "The B4 adapter groups both indexed release facts and records the related operator decision."
    )
    output, _ = _invoke_rejoin(fruit, _response(([0, 1, 2], composed)))
    node = _by_content(output, composed)

    assert node["category"] == "architecture"
    assert set(node["tags"]) == {"b4", "shared", "second", "operator"}
    assert len(node["tags"]) == 4


def test_category_tie_uses_the_first_member_in_the_model_group():
    fruit = _real_pipeline([
        _FactSpec("The architecture record names the indexed B4 adapter.", "architecture"),
        _FactSpec("The decision record enables the indexed B4 adapter.", "decision"),
    ])
    composed = "The decision enables the indexed B4 adapter named by the architecture record."
    output, _ = _invoke_rejoin(fruit, _response(([1, 0], composed)))

    assert _by_content(output, composed)["category"] == "decision"


def test_composition_decision_log_binds_members_sources_digests_and_content(tmp_path):
    fruit = _real_pipeline([
        _FactSpec("The B4 adapter records the first member's durable source identity.", tags=("audit",)),
        _FactSpec("The B4 adapter records the second member's content digest.", tags=("audit",)),
    ])
    composed = (
        "The B4 adapter records each member's durable source identity and content digest."
    )
    decision_path = tmp_path / "decisions.jsonl"
    _invoke_rejoin(
        fruit,
        _response(([0, 1], composed)),
        decision_log_path=decision_path,
    )

    entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["action"] == "b4-compose"
    assert entry["cluster_id"] == fruit.cluster_id
    assert entry["member_indices"] == [0, 1]
    assert entry["source_unit_ids"] == [
        item["source_unit_id"] for item in fruit.action_items
    ]
    assert entry["member_content_digests"] == [
        hashlib.sha256(item["content"].encode("utf-8")).hexdigest()
        for item in fruit.action_items
    ]
    assert entry["composed_content"] == composed


def test_reconcile_ignores_b2_source_unit_id_on_real_action_item(tmp_path):
    content = (
        "The sprint-41 B4 provenance probe carries source_unit_id through B2 while reconcile "
        "persists this concrete, project-specific durable node without changing its schema."
    )
    fruit = _real_pipeline([_FactSpec(content, "architecture", ("b4", "provenance"))])
    item = fruit.action_items[0]
    assert item["source_unit_id"] == fruit.envelopes[0].source_unit_id

    knowledge_path = tmp_path / "knowledge.jsonl"
    result = consolidate._apply_consolidation_result(
        {"nodes": [item]}, [], fruit.cluster, knowledge_path,
    )
    persisted = read_nodes(knowledge_path)
    assert result.nodes_created == 1
    assert len(persisted) == 1
    assert persisted[0].content == content


def test_composed_output_preserves_reconcile_shape_through_a_real_persist(tmp_path):
    fruit = _real_pipeline([
        _FactSpec(
            "The sprint-41 B4 shape probe starts with a concrete indexed member from extract_batch.",
            "architecture",
            ("b4",),
        ),
        _FactSpec(
            "The sprint-41 B4 shape probe sends the resulting compound through real reconcile.",
            "architecture",
            ("reconcile",),
        ),
    ])
    composed = (
        "The sprint-41 B4 shape probe starts with concrete indexed extract_batch members and "
        "sends their resulting compound through the real reconcile persistence boundary."
    )
    output, _ = _invoke_rejoin(fruit, _response(([0, 1], composed)))

    knowledge_path = tmp_path / "knowledge.jsonl"
    result = consolidate._apply_consolidation_result(
        {"nodes": output}, [], fruit.cluster, knowledge_path,
    )
    persisted = read_nodes(knowledge_path)
    assert result.nodes_created == 1
    assert len(persisted) == 1
    assert persisted[0].content == composed
    assert persisted[0].category == "architecture"
    assert set(persisted[0].tags) == {"b4", "reconcile"}


# Guard 4: temporal bounds are deterministic member metadata, never text re-detection.


def test_identical_member_bounds_carry_to_the_composed_node():
    fruit = _real_pipeline([
        _FactSpec(
            "The first rollout window runs from March 1 through April 30.",
            temporal_role="range",
            resolved="2025-03-01",
            resolved_end="2025-04-30",
        ),
        _FactSpec(
            "The second rollout condition uses the same March 1 through April 30 window.",
            temporal_role="range",
            resolved="2025-03-01",
            resolved_end="2025-04-30",
        ),
    ])
    composed = "Both rollout conditions use the March 1 through April 30 validity window."
    output, _ = _invoke_rejoin(fruit, _response(([0, 1], composed)))
    node = _by_content(output, composed)

    assert node["valid_from"] == "2025-03-01"
    assert node["valid_until"] == "2025-04-30"


def test_conflicting_non_null_bounds_split_back_to_original_bound_compatible_members():
    """A one-pass response has no source-clean text for post-hoc sub-compositions.

    Therefore a proposed cross-bound group splits precision-first to its original members. It
    must not reuse the all-member composition for either temporal partition or run inference a
    second time.
    """
    fruit = _real_pipeline([
        _FactSpec(
            "The 2025 signing key expires April 30.",
            temporal_role="expiry",
            resolved="2025-04-30",
        ),
        _FactSpec(
            "The 2026 signing key expires April 30.",
            temporal_role="expiry",
            resolved="2026-04-30",
        ),
    ])
    output, requests = _invoke_rejoin(
        fruit,
        _response(([0, 1], "Both signing keys expire April 30.")),
    )

    assert output == fruit.action_items
    assert len(requests) == 1


def test_non_overlapping_single_sided_bounds_split_instead_of_synthesizing_a_range():
    """Two attributed boundaries on different members do not jointly assert one validity range."""
    fruit = _real_pipeline([
        _FactSpec(
            "The hardened signing workflow becomes effective March 1.",
            temporal_role="effective",
            resolved="2025-03-01",
        ),
        _FactSpec(
            "The release candidate signing key expires April 30.",
            temporal_role="expiry",
            resolved="2025-04-30",
        ),
    ])
    output, requests = _invoke_rejoin(
        fruit,
        _response((
            [0, 1],
            "The hardened signing workflow and release candidate key are valid from March 1 "
            "through April 30.",
        )),
    )

    assert output == fruit.action_items
    assert len(requests) == 1


def test_mixed_null_and_single_non_null_member_carries_the_attributed_bound():
    fruit = _real_pipeline([
        _FactSpec(
            "The release candidate signing key expires April 30.",
            temporal_role="expiry",
            resolved="2025-04-30",
        ),
        _FactSpec("The release candidate uses the hardened signing workflow."),
        _FactSpec("The release candidate publishes its checksum manifest."),
    ])
    composed = (
        "The release candidate uses the hardened signing workflow, publishes its checksum "
        "manifest, and its signing key expires April 30."
    )
    output, _ = _invoke_rejoin(fruit, _response(([0, 1, 2], composed)))
    node = _by_content(output, composed)

    assert node["valid_from"] is None
    assert node["valid_until"] == "2025-04-30"


# Guard 5: any unusable compose response degrades the whole cluster loudly and losslessly.


@pytest.mark.parametrize(
    ("completion", "reason_fragments"),
    [
        ("not-json", ("parse", "json", "unparseable")),
        ({"wrong_key": []}, ("group", "shape", "schema")),
        (RuntimeError("compose backend unavailable"), ("unavailable", "infer", "exception", "backend")),
    ],
)
def test_failed_compose_response_fails_open_whole_cluster_with_loud_marker(
    truth_fruit, caplog, completion, reason_fragments,
):
    caplog.set_level(logging.WARNING)
    output, _ = _invoke_rejoin(truth_fruit, completion)

    assert output == truth_fruit.action_items
    messages = _warning_messages(caplog)
    assert any("B4_COMPOSE_FAIL_OPEN" in message for message in messages)
    assert any(truth_fruit.cluster_id in message for message in messages)
    assert any(
        any(fragment in message.lower() for fragment in reason_fragments)
        for message in messages
    )


# Stage mechanics: one scaled inference call, then B4's result enters B3.


def test_rejoin_budget_scales_with_create_count_instead_of_a_flat_800_floor():
    small = _real_pipeline([
        _FactSpec(f"The B4 budget probe small member {index} carries concrete recall metadata.")
        for index in range(2)
    ], cluster_id="b4-budget-small")
    large = _real_pipeline([
        _FactSpec(f"The B4 budget probe large member {index} carries concrete recall metadata.")
        for index in range(12)
    ], cluster_id="b4-budget-large")

    def singleton_response(items):
        return _response(*[
            ([index], item["content"]) for index, item in enumerate(items)
        ])

    _, small_requests = _invoke_rejoin(small, singleton_response(small.action_items))
    _, large_requests = _invoke_rejoin(large, singleton_response(large.action_items))

    assert len(small_requests) == 1 and len(large_requests) == 1
    assert large_requests[0]["max_tokens"] > small_requests[0]["max_tokens"]
    assert large_requests[0]["max_tokens"] > 800


class _RealPathClient:
    def __init__(self, specs: list[_FactSpec]):
        self.specs = specs

    def chat(self, *, messages, **_kwargs):
        prompt = messages[-1].content
        if "Extract structured data" in prompt:
            matches = [spec for spec in self.specs if spec.text in prompt]
            assert len(matches) == 1
            return _extraction_completion(matches[0])
        if "deciding how new facts relate" in prompt:
            return json.dumps({
                "actions": [
                    {"index": index, "action": "create", "tags": list(spec.tags)}
                    for index, spec in enumerate(self.specs)
                ]
            })
        raise AssertionError(f"unexpected unmocked model request: {prompt[:80]}")


def test_run_extract_path_inserts_b4_between_real_b2_output_and_b3(
    tmp_path, monkeypatch,
):
    specs = [
        _FactSpec("The real-path B4 wiring probe extracts the first indexed member.", tags=("b4",)),
        _FactSpec("The real-path B4 wiring probe extracts the second indexed member.", tags=("b4",)),
    ]
    cluster = _journal_cluster(specs)
    decision_path = tmp_path / "decisions.jsonl"
    seen: dict = {}
    rejoined = [
        {
            "action": "create",
            "existing_id": None,
            "content": (
                "The real-path B4 wiring probe carries both indexed extract members into "
                "the exact list passed to reconcile."
            ),
            "category": "fact",
            "tags": ["b4"],
            "source_turns": [],
            "contradiction_note": "",
            "valid_from": None,
            "valid_until": None,
        }
    ]

    def rejoin_spy(action_items, cluster_id, infer, *, decision_log_path=None):
        seen["b2_items"] = deepcopy(action_items)
        seen["cluster_id"] = cluster_id
        seen["infer"] = infer
        seen["decision_log_path"] = decision_log_path
        return deepcopy(rejoined)

    def apply_spy(parsed, *_args, **_kwargs):
        seen["b3_nodes"] = deepcopy(parsed["nodes"])
        return consolidate.ConsolidationResult()

    monkeypatch.setattr(consolidate, "_rejoin_create_actions", rejoin_spy, raising=False)
    monkeypatch.setattr(consolidate, "_apply_consolidation_result", apply_spy)

    result = consolidate._run_extract_path(
        cluster,
        "b4-wiring-real",
        _RealPathClient(specs),
        "fake-model",
        tmp_path / "failures.jsonl",
        [],
        tmp_path / "knowledge.jsonl",
        decision_log_path=decision_path,
    )

    assert result is not None
    assert seen["cluster_id"] == "b4-wiring-real"
    assert len(seen["b2_items"]) == 2
    assert all(item["source_unit_id"].startswith("b4-wiring-real:") for item in seen["b2_items"])
    assert seen["decision_log_path"] == decision_path
    assert seen["b3_nodes"] == rejoined
