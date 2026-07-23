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
from synapt.recall.content_profile import ContentProfile
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
    content_profile=None,
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
        content_profile=content_profile,
    )
    return output, requests


def _response(*groups: tuple[list[int], str]) -> dict:
    return {
        "groups": [
            {"indices": indices, "content": content}
            for indices, content in groups
        ]
    }


def _supersession_response(*groups: tuple[list[int], str, list[int]]) -> dict:
    return {
        "groups": [
            {"indices": indices, "content": content, "supersedes": supersedes}
            for indices, content, supersedes in groups
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
        "The sprint-41 B4 stage stays behind the SYNAPT_USE_EXTRACT flag and guards indexed "
        "member groups deterministically before reconcile executes."
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
        "The B4 adapter groups both indexed release facts together and records the related "
        "operator decision alongside them for the sprint-41 release."
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
    composed = (
        "The operator decision enables the indexed B4 adapter named by the architecture "
        "record for the full sprint-41 rollout window."
    )
    output, _ = _invoke_rejoin(fruit, _response(([1, 0], composed)))

    assert _by_content(output, composed)["category"] == "decision"


def test_composition_decision_log_binds_members_sources_digests_and_content(tmp_path):
    fruit = _real_pipeline([
        _FactSpec("The B4 adapter records the first member's durable source identity.", tags=("audit",)),
        _FactSpec("The B4 adapter records the second member's content digest.", tags=("audit",)),
    ])
    composed = (
        "The B4 adapter records each indexed member's durable source identity and content "
        "digest inside the sprint-41 decision log."
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


# Guard 6: same-batch supersession suppresses the older create and logs the decision.


def test_guard6_real_cache_suffix_supersession_suppresses_stale_create_and_logs_pair(
    tmp_path,
):
    """Mutation intent: sever guard 6 and the packet-real stale create survives.

    Both facts and source identities are the exact same-cluster pair from Atlas's frozen-B2
    replay, joined and locked by Apollo at config d7fa883. Config 8dcb818 closes chronological
    candidate ordering structurally, so this fixture consumes that guarantee without treating
    it as an open risk.
    """
    stale = "extract path does not share the legacy response_cache"
    current = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )
    fruit = _real_pipeline(
        [
            _FactSpec(stale, category="configuration"),
            _FactSpec(current, category="solution"),
        ],
        cluster_id="986d09c3e8bb2ae5",
    )
    action_items = deepcopy(fruit.action_items)
    stale_source = "986d09c3e8bb2ae5:0:decisions:5"
    current_source = "986d09c3e8bb2ae5:2:done:6"
    action_items[0]["source_unit_id"] = stale_source
    action_items[1]["source_unit_id"] = current_source
    decision_path = tmp_path / "decisions.jsonl"

    output, requests = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], stale, []),
            ([1], current, [0]),
        ),
        action_items=action_items,
        decision_log_path=decision_path,
    )

    assert [item["content"] for item in output] == [current]
    assert output[0]["source_unit_id"] == current_source

    entries = [
        json.loads(line) for line in decision_path.read_text().splitlines() if line
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["action"] == "b4-supersede"
    assert entry["cluster_id"] == fruit.cluster_id
    assert entry["superseded_index"] == 0
    assert entry["superseding_index"] == 1
    assert entry["superseded_source_unit_id"] == stale_source
    assert entry["superseding_source_unit_id"] == current_source
    assert (
        entry["superseded_content_digest"]
        == hashlib.sha256(stale.encode("utf-8")).hexdigest()
    )
    assert (
        entry["superseding_content_digest"]
        == hashlib.sha256(current.encode("utf-8")).hexdigest()
    )

    knowledge_path = tmp_path / "knowledge.jsonl"
    result = consolidate._apply_consolidation_result(
        {"nodes": output}, [], fruit.cluster, knowledge_path,
    )
    persisted = read_nodes(knowledge_path)
    assert result.nodes_created == 1
    assert [node.content for node in persisted if node.status == "active"] == [current]

    prompt = requests[0]["prompt"].lower()
    assert '"supersedes"' in prompt
    assert "same subject" in prompt
    assert all(root in prompt for root in ("updat", "correct", "contradict"))


def test_guard6_negation_flip_suppresses_earlier_ungrouped_singleton(tmp_path):
    """Mutation intent: trust model direction and the newer correction is suppressed.

    The fake deliberately reports the semantic pair from the earlier group's side. Guard 6
    must use the chronologically ordered candidate positions guaranteed by config 8dcb818,
    not the model's direction, and still suppress the earlier claim.
    """
    earlier = "the four attribution test failures are unrelated to the consolidate/extract change."
    later = "root-caused the four attribution test failures to the extract-path change after all."
    fruit = _real_pipeline(
        [
            _FactSpec(earlier, category="debugging"),
            _FactSpec(later, category="debugging"),
        ],
        cluster_id="guard6-negation-flip",
    )
    action_items = deepcopy(fruit.action_items)
    action_items[0]["source_unit_id"] = "s020c00:0"
    action_items[1]["source_unit_id"] = "s088c00:0"
    decision_path = tmp_path / "decisions.jsonl"

    output, _ = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], earlier, [1]),
            ([1], later, []),
        ),
        action_items=action_items,
        decision_log_path=decision_path,
    )

    assert [item["content"] for item in output] == [later]
    entries = [
        json.loads(line) for line in decision_path.read_text().splitlines() if line
    ]
    assert len(entries) == 1
    assert entries[0]["action"] == "b4-supersede"
    assert entries[0]["superseded_source_unit_id"] == "s020c00:0"
    assert entries[0]["superseding_source_unit_id"] == "s088c00:0"


def test_guard6_status_change_suppresses_open_state_and_keeps_closed_state(tmp_path):
    """Mutation intent: sever status-change suppression and both open and closed stay live."""
    opened = (
        "opened recall#900 to track the flat-budget truncation on the collection pass."
    )
    closed = "closed recall#900 \u2014 the dynamic-budget fix landed in the same PR."
    fruit = _real_pipeline(
        [
            _FactSpec(opened, category="action"),
            _FactSpec(closed, category="action"),
        ],
        cluster_id="guard6-status-change",
    )
    action_items = deepcopy(fruit.action_items)
    action_items[0]["source_unit_id"] = "s030c00:0"
    action_items[1]["source_unit_id"] = "s095c00:0"
    decision_path = tmp_path / "decisions.jsonl"

    output, _ = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], opened, []),
            ([1], closed, [0]),
        ),
        action_items=action_items,
        decision_log_path=decision_path,
    )

    assert [item["content"] for item in output] == [closed]
    entries = [
        json.loads(line) for line in decision_path.read_text().splitlines() if line
    ]
    assert len(entries) == 1
    assert entries[0]["action"] == "b4-supersede"
    assert entries[0]["superseded_index"] == 0
    assert entries[0]["superseding_index"] == 1


def test_guard6_shared_vocabulary_without_conflict_keeps_both_creates(tmp_path):
    """Mutation intent: replace the semantic edge with vocabulary matching and this turns RED."""
    derivation = (
        "the response_cache key is a sha256 of sorted session_id|timestamp pairs."
    )
    suffix = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )
    fruit = _real_pipeline(
        [
            _FactSpec(derivation, category="architecture"),
            _FactSpec(suffix, category="solution"),
        ],
        cluster_id="guard6-shared-vocabulary-control",
    )
    action_items = deepcopy(fruit.action_items)
    action_items[0]["source_unit_id"] = "s060c00:0"
    action_items[1]["source_unit_id"] = "986d09c3e8bb2ae5:2:done:6"
    decision_path = tmp_path / "decisions.jsonl"

    output, requests = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], derivation, []),
            ([1], suffix, []),
        ),
        action_items=action_items,
        decision_log_path=decision_path,
    )

    assert output == action_items
    assert not decision_path.exists()
    prompt = requests[0]["prompt"].lower()
    assert "shared vocabulary" in prompt
    assert "supersedes" in prompt


# Guard 6 (A3) supplemental coverage: real behaviors the 4 A2-pinned fixtures don't exercise
# (all 4 use single-digit entry_index values, always-surviving singletons, and well-formed
# supersedes fields), but that are load-bearing for the implementation itself.


def test_guard6_malformed_supersedes_field_type_degrades_to_no_claim(tmp_path):
    """Mutation intent, two distinct guards in _b4_validate_compose_response, verified
    separately rather than assumed:

    The OUTER isinstance(raw_supersedes, list) check protects against a non-list
    supersedes value. Both a string ("0,1", Opus's A3-review example) and a bare int (5) hit
    this guard — checked directly: a Python str is NOT a list instance, so the string case
    never even reaches the inner per-element loop; disproved my own first-draft assumption
    that it would "iterate as characters" before writing this final version.

    The INNER isinstance(i, int)/not-bool/range check protects against a well-formed LIST
    whose ELEMENTS are the wrong shape (["not-an-int"]) — the only case that actually reaches
    that line; the outer guard alone does not cover it.

    Composition is preserved in every case (never costs data) — but per Sentinel's spec-author
    ruling (m_4adb43c0), a PRESENT-but-malformed field is a real detection-signal failure, not
    a benign absence, and must emit exactly one b4-group-rejected/malformed-supersedes entry.
    """
    stale = "extract path does not share the legacy response_cache"
    current = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )

    for case_index, malformed_value in enumerate(("0,1", 5, ["not-an-int"])):
        fruit = _real_pipeline(
            [
                _FactSpec(stale, category="configuration"),
                _FactSpec(current, category="solution"),
            ],
            cluster_id=f"guard6-malformed-supersedes-{case_index}",
        )
        decision_path = tmp_path / f"decisions-{case_index}.jsonl"

        malformed_response = {
            "groups": [
                {"indices": [0], "content": stale, "supersedes": malformed_value},
                {"indices": [1], "content": current, "supersedes": []},
            ],
        }

        output, _ = _invoke_rejoin(fruit, malformed_response, decision_log_path=decision_path)

        assert len(output) == 2, f"case {case_index} ({malformed_value!r}) dropped an item"
        assert {item["content"] for item in output} == {stale, current}

        entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
        rejections = [e for e in entries if e["action"] == "b4-group-rejected"]
        assert len(rejections) == 1, f"case {case_index} ({malformed_value!r})"
        assert rejections[0]["stage"] == "malformed-supersedes"
        assert rejections[0]["member_indices"] == [0]

        reason = rejections[0]["reason"].lower()
        if isinstance(malformed_value, list):
            # ["not-an-int"] IS a list — it hits the per-element filter, not the outer
            # not-a-list guard, so it carries the partial-invalid reason contract.
            assert "filtered" in reason, f"case {case_index}: {reason!r}"
            assert "valid targets retained" in reason, f"case {case_index}: {reason!r}"
        else:
            assert "ignored" in reason, f"case {case_index}: {reason!r}"
        assert "composition preserved" in reason, f"case {case_index}: {reason!r}"

        knowledge_path = tmp_path / f"knowledge-{case_index}.jsonl"
        result = consolidate._apply_consolidation_result(
            {"nodes": output}, [], fruit.cluster, knowledge_path,
        )
        assert result.nodes_created == 2


def test_guard6_missing_supersedes_field_is_silent_no_telemetry(tmp_path):
    """A MISSING supersedes field (vs. a present-but-malformed one) degrades silently — no
    telemetry — for backward compatibility, per Sentinel's spec-author ruling: the field didn't
    exist before this sprint, so its absence is not a detection-signal failure."""
    stale = "extract path does not share the legacy response_cache"
    current = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )
    fruit = _real_pipeline(
        [
            _FactSpec(stale, category="configuration"),
            _FactSpec(current, category="solution"),
        ],
        cluster_id="guard6-missing-supersedes-field",
    )
    decision_path = tmp_path / "decisions.jsonl"

    response_without_supersedes_key = {
        "groups": [
            {"indices": [0], "content": stale},
            {"indices": [1], "content": current},
        ],
    }

    output, _ = _invoke_rejoin(
        fruit, response_without_supersedes_key, decision_log_path=decision_path,
    )

    assert len(output) == 2
    assert {item["content"] for item in output} == {stale, current}
    assert not decision_path.exists()


def test_guard6_out_of_range_supersedes_target_is_filtered_not_a_wrong_suppression(tmp_path):
    """A hallucinated target index (999, out of range for a 2-item cluster) must not crash
    creates[target] or silently suppress the wrong thing, AND must emit malformed-supersedes
    telemetry. Mutation-verified, corrected after Sentinel's live-diff note: an earlier version
    of this docstring claimed removing the upstream `0 <= i < n` range filter would NOT turn
    this test red, reasoning that _surviving_position's member_to_group_first lookup
    independently no-ops on 999 regardless. That was true for suppression-avoidance alone, but
    is STALE now that this test also asserts the malformed-supersedes telemetry entry — which
    depends directly on the range filter (without it, len(supersedes) == len(raw_supersedes)
    and no malformed_reason is produced). Re-verified directly: removing the filter now DOES
    turn this test red."""
    stale = "extract path does not share the legacy response_cache"
    current = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )
    fruit = _real_pipeline(
        [
            _FactSpec(stale, category="configuration"),
            _FactSpec(current, category="solution"),
        ],
        cluster_id="guard6-out-of-range-supersedes-target",
    )
    decision_path = tmp_path / "decisions.jsonl"

    output, _ = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], stale, []),
            ([1], current, [999]),  # hallucinated target, out of range for n=2
        ),
        decision_log_path=decision_path,
    )

    assert len(output) == 2
    assert {item["content"] for item in output} == {stale, current}

    entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
    rejections = [e for e in entries if e["action"] == "b4-group-rejected"]
    assert len(rejections) == 1
    assert rejections[0]["stage"] == "malformed-supersedes"
    assert rejections[0]["member_indices"] == [1]
    reason = rejections[0]["reason"].lower()
    assert "filtered" in reason
    assert "valid targets retained" in reason
    assert "composition preserved" in reason


def test_guard6_mixed_valid_and_out_of_range_supersedes_targets_retains_the_valid_one(tmp_path):
    """A supersedes list with BOTH a valid target and a hallucinated one must not be discarded
    wholesale — the valid target's claim still fires (Sentinel: "malformed optional detection
    metadata must never cost data"), AND the malformed-supersedes telemetry still records that
    the list was partially invalid."""
    stale = "extract path does not share the legacy response_cache"
    current = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )
    fruit = _real_pipeline(
        [
            _FactSpec(stale, category="configuration"),
            _FactSpec(current, category="solution"),
        ],
        cluster_id="guard6-mixed-valid-and-out-of-range",
    )
    action_items = deepcopy(fruit.action_items)
    action_items[0]["source_unit_id"] = "986d09c3e8bb2ae5:0:decisions:5"
    action_items[1]["source_unit_id"] = "986d09c3e8bb2ae5:2:done:6"
    decision_path = tmp_path / "decisions.jsonl"

    output, _ = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], stale, []),
            ([1], current, [0, 999]),  # 0 is a real, valid target; 999 is hallucinated
        ),
        action_items=action_items,
        decision_log_path=decision_path,
    )

    # The valid target (0) still fires: stale is suppressed, current survives.
    assert [item["content"] for item in output] == [current]

    entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
    supersede_entries = [e for e in entries if e["action"] == "b4-supersede"]
    malformed_entries = [e for e in entries if e["action"] == "b4-group-rejected"]
    assert len(supersede_entries) == 1
    assert supersede_entries[0]["superseded_index"] == 0
    assert supersede_entries[0]["superseding_index"] == 1
    assert len(malformed_entries) == 1
    assert malformed_entries[0]["stage"] == "malformed-supersedes"
    assert malformed_entries[0]["member_indices"] == [1]
    reason = malformed_entries[0]["reason"].lower()
    assert "filtered" in reason
    assert "valid targets retained" in reason
    assert "composition preserved" in reason


def test_guard6_same_entry_chronology_tie_preserves_both_and_logs_ambiguous_order(tmp_path):
    """A1 §7's documented caveat, pinned end-to-end (Sentinel, A3 re-review blocker 1): two
    facts sharing the SAME source_unit_id entry_index carry no usable direction signal. The
    original boolean-only comparison returned False for BOTH directions on a tie, and the
    caller's if/else silently treated "not earlier" as "later" — suppressing whichever side
    happened to be the loop's `target`. Both candidates must survive; the claim is recorded as
    unresolved, not silently discarded."""
    earlier_field = "extract path does not share the legacy response_cache"
    later_field = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )
    fruit = _real_pipeline(
        [
            _FactSpec(earlier_field, category="configuration"),
            _FactSpec(later_field, category="solution"),
        ],
        cluster_id="guard6-same-entry-tie",
    )
    action_items = deepcopy(fruit.action_items)
    action_items[0]["source_unit_id"] = "986d09c3e8bb2ae5:0:decisions:4"
    action_items[1]["source_unit_id"] = "986d09c3e8bb2ae5:0:decisions:5"  # same entry_index
    decision_path = tmp_path / "decisions.jsonl"

    output, _ = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], earlier_field, []),
            ([1], later_field, [0]),
        ),
        action_items=action_items,
        decision_log_path=decision_path,
    )

    assert {item["content"] for item in output} == {earlier_field, later_field}

    entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
    assert not any(e["action"] == "b4-supersede" for e in entries)
    rejections = [e for e in entries if e["action"] == "b4-group-rejected"]
    assert len(rejections) == 1
    assert rejections[0]["stage"] == "ambiguous-supersession-order"
    assert set(rejections[0]["member_indices"]) == {0, 1}


def test_guard6_decision_log_write_failure_keeps_both_candidates(tmp_path):
    """Sentinel, A3 re-review blocker 2: suppression is only truth-preserving when the audit
    record is durable. A write failure must NOT suppress — both candidates stay in output,
    exactly as if no supersession claim had ever been made."""
    stale = "extract path does not share the legacy response_cache"
    current = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )
    fruit = _real_pipeline(
        [
            _FactSpec(stale, category="configuration"),
            _FactSpec(current, category="solution"),
        ],
        cluster_id="guard6-decision-log-write-failure",
    )
    action_items = deepcopy(fruit.action_items)
    action_items[0]["source_unit_id"] = "986d09c3e8bb2ae5:0:decisions:5"
    action_items[1]["source_unit_id"] = "986d09c3e8bb2ae5:2:done:6"

    # decision_log_path's PARENT already exists as a FILE (not a directory), so the real
    # mkdir(parents=True) inside _log_b4_supersede_decision raises OSError — a genuine write
    # failure, not a mocked one (same technique as guard 3's own write-failure test).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    unwritable_log = blocker / "decisions.jsonl"

    direct_logged_ok = consolidate._log_b4_supersede_decision(
        unwritable_log, "guard6-decision-log-write-failure", 0, 1,
        "986d09c3e8bb2ae5:0:decisions:5", "986d09c3e8bb2ae5:2:done:6",
        stale, current,
    )
    assert direct_logged_ok is False

    output, _ = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], stale, []),
            ([1], current, [0]),
        ),
        action_items=action_items,
        decision_log_path=unwritable_log,
    )

    assert {item["content"] for item in output} == {stale, current}
    assert not unwritable_log.exists()


def test_guard6_decision_log_path_none_keeps_both_candidates(tmp_path):
    """Sentinel, A3 audit clarification: decision_log_path=None is NOT a free pass for
    suppression, unlike guard 3's compose-provenance opt-out — supersession deletes a claim, so
    with no path to record why, both candidates must stay."""
    stale = "extract path does not share the legacy response_cache"
    current = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )
    fruit = _real_pipeline(
        [
            _FactSpec(stale, category="configuration"),
            _FactSpec(current, category="solution"),
        ],
        cluster_id="guard6-decision-log-path-none",
    )
    action_items = deepcopy(fruit.action_items)
    action_items[0]["source_unit_id"] = "986d09c3e8bb2ae5:0:decisions:5"
    action_items[1]["source_unit_id"] = "986d09c3e8bb2ae5:2:done:6"

    assert consolidate._log_b4_supersede_decision(
        None, "x", 0, 1, "a", "b", stale, current,
    ) is False

    output, _ = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], stale, []),
            ([1], current, [0]),
        ),
        action_items=action_items,
        decision_log_path=None,
    )

    assert {item["content"] for item in output} == {stale, current}


def test_guard6_accepted_composition_member_mapping_is_consistent_by_index(tmp_path):
    """Sentinel, A3 re-review blocker 3 + test-correction: for accepted composition [0,1], the
    real current/suffix singleton targeting EITHER member's index must produce the SAME
    verdict — supersedes is defined over any candidate index, and both members now refer to one
    logical memory. The original implementation silently no-op'd when the target was the
    non-first member but suppressed the whole composed output when the target was the first
    member, purely because of an implementation artifact (only the first member had a mapped
    position).

    Uses the REAL stale cache-claim plus its real prompt-shape sibling context as the composed
    group (genuinely related content — Sentinel's correction: a fabricated edge between
    semantically UNRELATED facts, as an earlier draft of this test used, violates A2's truth
    contract even for a position-mechanics test), with member ORDER swapped across the two
    cases so the stale claim sits at position 0 in one and position 1 in the other. The real
    current/suffix singleton's supersedes claim always targets the stale claim's OWN index,
    whichever position it lands in — asserting exactly one b4-supersede entry with
    superseded_index equal to that same index."""
    stale = "extract path does not share the legacy response_cache"
    prompt_shape_context = "different prompt shapes"
    current = (
        'writing extract-path results to response_cache under a distinct ":extract" '
        "key suffix"
    )
    composed_content = (
        "The extract path does not share the legacy response_cache, and the extract and "
        "legacy paths use different prompt shapes."
    )

    for stale_index in (0, 1):
        member_order = (
            [stale, prompt_shape_context] if stale_index == 0
            else [prompt_shape_context, stale]
        )
        fruit = _real_pipeline(
            [
                _FactSpec(member_order[0], category="configuration"),
                _FactSpec(member_order[1], category="reason"),
                _FactSpec(current, category="solution"),
            ],
            cluster_id=f"guard6-composed-member-mapping-stale-at-{stale_index}",
        )
        action_items = deepcopy(fruit.action_items)
        action_items[0]["source_unit_id"] = "986d09c3e8bb2ae5:0:decisions:4"
        action_items[1]["source_unit_id"] = "986d09c3e8bb2ae5:0:decisions:5"
        action_items[2]["source_unit_id"] = "986d09c3e8bb2ae5:2:done:6"
        decision_path = tmp_path / f"decisions-stale-at-{stale_index}.jsonl"

        output, _ = _invoke_rejoin(
            fruit,
            _supersession_response(
                ([0, 1], composed_content, []),
                ([2], current, [stale_index]),  # always targets the stale claim's own index
            ),
            action_items=action_items,
            decision_log_path=decision_path,
        )

        # The current/suffix singleton is chronologically LATER than the composed group
        # (entry_index 2 > 0), so it supersedes the composition regardless of which original
        # member index held the stale claim.
        assert [item["content"] for item in output] == [current], f"stale_index={stale_index}"

        entries = [
            json.loads(line) for line in decision_path.read_text().splitlines() if line
        ]
        supersede_entries = [e for e in entries if e["action"] == "b4-supersede"]
        assert len(supersede_entries) == 1, f"stale_index={stale_index}"
        assert supersede_entries[0]["superseded_index"] == stale_index
        assert supersede_entries[0]["superseding_index"] == 2

    # The self-reference case: the composed group's OWN claim against its own sibling member.
    # DISTINCT entry_index values on the two members (Sentinel, mutation-mask catch): with the
    # default _real_pipeline source_unit_ids, both members share entry_index 0, so the tie
    # guard (order == 0) would ALSO independently preserve both candidates — masking whether
    # the same-output guard (rep_pos == target_pos) is doing anything at all. Distinct indices
    # here mean the tie guard can't fire, isolating the same-output guard genuinely.
    fruit = _real_pipeline(
        [
            _FactSpec(stale, category="configuration"),
            _FactSpec(prompt_shape_context, category="reason"),
        ],
        cluster_id="guard6-composed-self-reference",
    )
    action_items = deepcopy(fruit.action_items)
    action_items[0]["source_unit_id"] = "986d09c3e8bb2ae5:0:decisions:4"
    action_items[1]["source_unit_id"] = "986d09c3e8bb2ae5:1:decisions:5"
    decision_path = tmp_path / "decisions-self-reference.jsonl"

    output, _ = _invoke_rejoin(
        fruit,
        _supersession_response(([0, 1], composed_content, [1])),
        action_items=action_items,
        decision_log_path=decision_path,
    )

    assert [item["content"] for item in output] == [composed_content]
    entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
    compose_entries = [e for e in entries if e["action"] == "b4-compose"]
    supersede_entries = [e for e in entries if e["action"] == "b4-supersede"]
    assert len(compose_entries) == 1
    assert len(supersede_entries) == 0


def test_source_unit_id_direction_uses_numeric_not_lexicographic_comparison():
    """Real-format entry_index parses as an int, not a lexicographically-sorted string —
    protects against inversion if _split_large_cluster's max_size default ever climbs past 9
    (Opus, 2026-07-16): "10" sorts before "2" lexicographically but is chronologically later.
    None of A2's 4 pinned fixtures reach double-digit entry_index (today's real max_size=4
    caps it at a single digit), so this is the only test that actually exercises the numeric
    branch rather than the lexicographic fallback."""
    earlier = "some-cluster-hash:2:decisions:0"
    later = "some-cluster-hash:10:decisions:0"
    assert consolidate._source_unit_id_order(earlier, later) == -1
    assert consolidate._source_unit_id_order(later, earlier) == 1


def test_source_unit_id_direction_falls_back_to_lexicographic_for_non_real_format():
    """A2's own fixtures (s020c00:0-style, 2 segments not 4) — and any other non-conforming
    source_unit_id shape — degrade to full-string comparison rather than crashing."""
    assert consolidate._source_unit_id_order("s020c00:0", "s088c00:0") == -1
    assert consolidate._source_unit_id_order("s088c00:0", "s020c00:0") == 1


def test_source_unit_id_direction_returns_tie_for_identical_or_same_entry_index():
    """A genuine tie (Sentinel, A3 re-review blocker 1) is a distinct outcome, 0, never
    silently resolved to a direction — same entry_index in the real format, or identical
    strings in the fallback shape."""
    assert consolidate._source_unit_id_order(
        "cluster:0:decisions:4", "cluster:0:decisions:5",
    ) == 0
    assert consolidate._source_unit_id_order("s020c00:0", "s020c00:0") == 0


def test_guard6_claim_does_not_apply_when_the_superseding_side_is_itself_rejected(tmp_path):
    """The superseding side must independently survive to real output before its claim can
    suppress anything — a multi-member group's content-safety rejection must not ALSO take
    down the singleton it claimed to supersede (never-drop applies to guard 6 too)."""
    stale = "extract path does not share the legacy response_cache"
    fruit = _real_pipeline(
        [
            _FactSpec(stale, category="configuration"),
            _FactSpec(
                "Recall stores durable node revisions in an append-versioned "
                "knowledge.jsonl file that survives process restarts without data loss.",
            ),
            _FactSpec(
                "The adapter failures reported in this cluster are unrelated to SQLite "
                "locking behavior observed during the retry-boundary investigation.",
            ),
        ],
        cluster_id="guard6-superseding-side-rejected",
    )
    decision_path = tmp_path / "decisions.jsonl"

    output, _ = _invoke_rejoin(
        fruit,
        _supersession_response(
            ([0], stale, []),
            ([1, 2], "   ", [0]),  # whitespace-only -> content-unsafe rejection
        ),
        decision_log_path=decision_path,
    )

    # The rejected group's members revert to individual pass-through, AND the claim they
    # carried never fires — stale must still be present, not suppressed on a claim that
    # never independently resolved to real output. The whitespace group's OWN content-unsafe
    # rejection is still logged (A4's existing guard-4 telemetry) — that's a different,
    # legitimate entry; the check here is that no b4-supersede entry exists alongside it.
    assert len(output) == 3
    assert any(item["content"] == stale for item in output)
    entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
    assert not any(e["action"] == "b4-supersede" for e in entries)


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

    def rejoin_spy(action_items, cluster_id, infer, *, decision_log_path=None, content_profile=None):
        seen["b2_items"] = deepcopy(action_items)
        seen["cluster_id"] = cluster_id
        seen["infer"] = infer
        seen["decision_log_path"] = decision_log_path
        seen["content_profile"] = content_profile
        return deepcopy(rejoined)

    def apply_spy(parsed, *_args, **_kwargs):
        seen["b3_nodes"] = deepcopy(parsed["nodes"])
        return consolidate.ConsolidationResult()

    monkeypatch.setattr(consolidate, "_rejoin_create_actions", rejoin_spy, raising=False)
    monkeypatch.setattr(consolidate, "_apply_consolidation_result", apply_spy)

    real_profile = ContentProfile(_type="mixed")
    result = consolidate._run_extract_path(
        cluster,
        "b4-wiring-real",
        _RealPathClient(specs),
        "fake-model",
        tmp_path / "failures.jsonl",
        [],
        tmp_path / "knowledge.jsonl",
        decision_log_path=decision_path,
        content_profile=real_profile,
    )

    assert result is not None
    assert seen["cluster_id"] == "b4-wiring-real"
    assert len(seen["b2_items"]) == 2
    assert all(item["source_unit_id"].startswith("b4-wiring-real:") for item in seen["b2_items"])
    assert seen["decision_log_path"] == decision_path
    # Identity, not just equality — proves _run_extract_path threads the SAME content_profile
    # object into _rejoin_create_actions, not a coincidentally-equal one (Opus, recall#884
    # reviewer-1 re-verify: the spy already captured this field but nothing asserted on it —
    # deleting the content_profile=content_profile kwarg at the real call site left the whole
    # suite green).
    assert seen["content_profile"] is real_profile
    assert seen["b3_nodes"] == rejoined


# Guard closures — Sentinel's recall#884 re-review (2026-07-15), all four fruit-reproduced.


def test_whitespace_composed_content_rejects_the_group_and_persists_original_members(tmp_path):
    """A whitespace-only composition is silently DROPPED by real _apply_consolidation_result
    (0 nodes created — verified against the actual create branch, not assumed). B4 must catch
    this before B3 ever sees it, or both members are lost."""
    fruit = _real_pipeline([
        _FactSpec(
            "The adapter failures reported in this cluster are unrelated to SQLite "
            "locking behavior observed during the retry-boundary investigation."
        ),
        _FactSpec(
            "Recall stores durable node revisions in an append-versioned "
            "knowledge.jsonl file that survives process restarts without data loss."
        ),
    ])
    output, _ = _invoke_rejoin(fruit, _response(([0, 1], "   ")))

    assert output == fruit.action_items

    knowledge_path = tmp_path / "knowledge.jsonl"
    result = consolidate._apply_consolidation_result(
        {"nodes": output}, [], fruit.cluster, knowledge_path,
    )
    assert result.nodes_created == 2
    assert len(read_nodes(knowledge_path)) == 2


def test_composed_content_over_b3_ceiling_rejects_the_group_instead_of_silently_truncating(
    tmp_path,
):
    """A composition over 300 chars is silently WORD-TRUNCATED by real
    _apply_consolidation_result (verified: a 339-char input persists at 298 chars, the final
    clause gone with no warning). B4 must reject it before that happens."""
    fruit = _real_pipeline([
        _FactSpec(
            "The adapter failures reported in this cluster are unrelated to SQLite "
            "locking behavior observed during the retry-boundary investigation."
        ),
        _FactSpec(
            "Recall stores durable node revisions in an append-versioned "
            "knowledge.jsonl file that survives process restarts without data loss."
        ),
    ])
    over_limit = "x" * (consolidate._B4_COMPOSE_CONTENT_MAX_CHARS + 1)
    output, _ = _invoke_rejoin(fruit, _response(([0, 1], over_limit)))

    assert output == fruit.action_items

    knowledge_path = tmp_path / "knowledge.jsonl"
    result = consolidate._apply_consolidation_result(
        {"nodes": output}, [], fruit.cluster, knowledge_path,
    )
    assert result.nodes_created == 2
    assert len(read_nodes(knowledge_path)) == 2


def test_low_specificity_composed_content_rejects_the_group_and_persists_original_members(
    tmp_path,
):
    """A short, NON-whitespace, well-under-300-char composed sentence can still silently
    vanish at real B3: _apply_consolidation_result's create branch also runs
    _is_generic_node/_lacks_specificity/contamination/_is_garbled_content, not just the
    empty/oversize checks. Discovered via adversarial verification (not Sentinel's original
    finding, but the same failure CLASS): "The build finished and all tests passed without
    any errors." (62 chars) sails past a length-only guard, then _lacks_specificity(...,
    threshold=120) drops it at B3 with nodes_created=0 — both members lost, same symptom."""
    fruit = _real_pipeline([
        _FactSpec(
            "The adapter failures reported in this cluster are unrelated to SQLite "
            "locking behavior observed during the retry-boundary investigation."
        ),
        _FactSpec(
            "Recall stores durable node revisions in an append-versioned "
            "knowledge.jsonl file that survives process restarts without data loss."
        ),
    ])
    low_specificity = "The build finished and all tests passed without any errors."
    output, _ = _invoke_rejoin(fruit, _response(([0, 1], low_specificity)))

    assert output == fruit.action_items

    knowledge_path = tmp_path / "knowledge.jsonl"
    result = consolidate._apply_consolidation_result(
        {"nodes": output}, [], fruit.cluster, knowledge_path,
    )
    assert result.nodes_created == 2
    assert len(read_nodes(knowledge_path)) == 2


def test_mixed_profile_specificity_threshold_rejects_the_group_and_persists_original_members(
    tmp_path,
):
    """A composed sentence can pass B4's no-profile check (well over 120 chars) yet still be
    dropped by REAL B3 once the actual production content_profile is threaded through — B3's
    mixed profile relaxes the specificity threshold to 200, not B4's old hard-coded 120
    (Sentinel, recall#884 re-review round 2, fruit 1: this exact composition passes with no
    profile, then real _apply_consolidation_result under a mixed profile creates 0, persists
    0 — both members gone, and the decision log would have claimed a composition that never
    persisted). The two original member facts are deliberately kept over 200 chars (the
    mixed profile's OWN threshold) so this test isolates the composed-content rejection —
    without that length margin, the mixed profile's relaxed-but-still-real specificity check
    can reject the pass-through ORIGINAL members too, which is a fixture-quality concern, not
    the mechanism this test targets."""
    fruit = _real_pipeline([
        _FactSpec(
            "The adapter failures reported in this cluster are unrelated to SQLite "
            "locking behavior observed during the retry-boundary investigation, "
            "confirmed independently across two separate dogfood runs this sprint."
        ),
        _FactSpec(
            "Recall stores durable node revisions in an append-versioned "
            "knowledge.jsonl file that survives process restarts without data loss, "
            "verified independently across two separate dogfood runs this sprint."
        ),
    ])
    mixed_profile = ContentProfile(_type="mixed")
    borderline = (
        "The build finished successfully and every configured test suite in the "
        "pipeline passed without reporting any errors or warnings along the way."
    )
    # Sanity-pin the exact threshold gap this test exercises, against the real functions.
    assert not consolidate._lacks_specificity(borderline, threshold=120, content_type=None)
    assert consolidate._lacks_specificity(borderline, threshold=200, content_type="mixed")

    output, _ = _invoke_rejoin(
        fruit, _response(([0, 1], borderline)), content_profile=mixed_profile,
    )

    assert output == fruit.action_items

    knowledge_path = tmp_path / "knowledge.jsonl"
    result = consolidate._apply_consolidation_result(
        {"nodes": output}, [], fruit.cluster, knowledge_path, content_profile=mixed_profile,
    )
    assert result.nodes_created == 2
    assert len(read_nodes(knowledge_path)) == 2


def test_section_prefix_normalization_rejects_the_group_and_persists_original_members(
    tmp_path,
):
    """A composed sentence that reads as specific with its section-header prefix intact can
    strip down to a low-specificity remainder — B4 must normalize (scrub/markdown/section-
    prefix-strip) BEFORE checking specificity, in the same order real B3 does, not check the
    raw string (Sentinel, recall#884 re-review round 2, fruit 2: a topic-prefixed composition
    passed B4 raw, then real B3 stripped the prefix and dropped the low-specificity remainder
    — created 0, persisted 0; no content_profile needed to reproduce this one)."""
    fruit = _real_pipeline([
        _FactSpec(
            "The adapter failures reported in this cluster are unrelated to SQLite "
            "locking behavior observed during the retry-boundary investigation."
        ),
        _FactSpec(
            "Recall stores durable node revisions in an append-versioned "
            "knowledge.jsonl file that survives process restarts without data loss."
        ),
    ])
    prefixed = (
        "Operational Deployment Readiness: The build finished and all configured "
        "tests passed without reporting any errors."
    )
    # Sanity-pin: raw passes, but the REAL stripped remainder (what B3 actually evaluates)
    # lacks specificity — against the real functions, not asserted from prose.
    assert not consolidate._lacks_specificity(prefixed, threshold=120, content_type=None)
    stripped = consolidate._strip_section_prefix(prefixed)
    assert stripped != prefixed
    assert consolidate._lacks_specificity(stripped, threshold=120, content_type=None)

    output, _ = _invoke_rejoin(fruit, _response(([0, 1], prefixed)))

    assert output == fruit.action_items

    knowledge_path = tmp_path / "knowledge.jsonl"
    result = consolidate._apply_consolidation_result(
        {"nodes": output}, [], fruit.cluster, knowledge_path,
    )
    assert result.nodes_created == 2
    assert len(read_nodes(knowledge_path)) == 2


def test_duplicate_membership_across_a_valid_and_an_invalid_group_still_fails_open(
    truth_fruit, caplog,
):
    """A real index appearing in a DROPPED (invalid-index-containing) group AND a separately
    valid group is still cross-group duplicate membership — the model's own bookkeeping is
    incoherent even though one of the two groups also happens to be invalid. Filtering the
    invalid group out before scanning for duplicates would hide this (fruit-confirmed,
    Sentinel: [0,1,999] + [0,2] let index 0 slip through under the old ordering)."""
    caplog.set_level(logging.WARNING)
    output, _ = _invoke_rejoin(
        truth_fruit,
        _response(
            ([0, 1, 999], "corrupted mixed group"),
            ([0, 2], "otherwise-valid group"),
        ),
    )

    assert output == truth_fruit.action_items
    messages = _warning_messages(caplog)
    assert any("B4_COMPOSE_FAIL_OPEN" in message for message in messages)
    assert any("duplicate" in message.lower() for message in messages)


def test_malformed_group_indices_type_fails_open_the_whole_cluster_with_loud_marker(
    truth_fruit, caplog,
):
    """A group whose ``indices`` is the WRONG TYPE (a string, not a list) is a schema
    violation, not an intentional unaddressed response — it must fail open loudly, not
    silently vanish via the same per-element skip that handles a genuinely absent index
    (fruit-confirmed, Sentinel: this shape produced zero B4_COMPOSE_FAIL_OPEN warnings under
    the old per-element ``continue``)."""
    caplog.set_level(logging.WARNING)
    output, _ = _invoke_rejoin(
        truth_fruit,
        {"groups": [{"indices": "0,1", "content": "joined"}]},
    )

    assert output == truth_fruit.action_items
    messages = _warning_messages(caplog)
    assert any("B4_COMPOSE_FAIL_OPEN" in message for message in messages)


def test_non_string_tag_element_is_sanitized_not_crashed():
    """A real B2 response can carry a non-string tag element (int 7 alongside "good") — a bare
    ``sorted({...})`` over the unioned tag set raises TypeError on a mixed str/int comparison,
    crashing B4 entirely OUTSIDE the fail-open path (fruit-confirmed, Sentinel). Tag elements
    must be sanitized the same way _apply_consolidation_result's own monolith-path tags are
    (``scrub_text(str(t)) for t in tags if t``) before the union, so this can never crash."""
    fruit = _real_pipeline([
        _FactSpec(
            "The B4 tag-safety probe records the first member with a numeric tag element.",
            tags=("good", 7),
        ),
        _FactSpec(
            "The B4 tag-safety probe records the second member with only string tags.",
            tags=("clean",),
        ),
    ])
    composed = (
        "The B4 tag-safety probe composes both indexed members together without crashing "
        "on the numeric tag element during sprint-41."
    )
    output, _ = _invoke_rejoin(fruit, _response(([0, 1], composed)))

    node = _by_content(output, composed)
    assert set(node["tags"]) == {"good", "7", "clean"}


def test_decision_log_write_failure_rejects_the_composition_not_an_untraceable_persist(
    tmp_path,
):
    """Guard 3 chose the decision log as v1's ONLY durable member provenance. A write failure
    that gets silently swallowed would persist a compound node with NO record anywhere of what
    it was composed from (fruit-confirmed, Sentinel). B4 must reject the composition instead —
    members fall back to individual pass-through, which at least stays traceable as atoms.

    Composed content is deliberately kept over B3's real 120-char specificity floor (Opus,
    recall#884 re-review: an earlier 80-char version of this string was itself rejected by
    the content-safety guard BEFORE _log_b4_compose_decision was ever called — the two guards
    shadowed each other, so the test passed for the wrong reason and a mutated
    except-OSError-return-True never turned it red). The direct call below additionally pins
    that the OSError genuinely fires INSIDE _log_b4_compose_decision itself, independent of
    _rejoin_create_actions, closing that seam for good."""
    fruit = _real_pipeline([
        _FactSpec("The B4 provenance-failure probe records the first member's identity."),
        _FactSpec("The B4 provenance-failure probe records the second member's identity."),
    ])
    composed = (
        "The B4 provenance-failure probe composes both indexed members' durable "
        "identities into one compound node during the sprint-41 decision-log gate."
    )
    # decision_log_path's PARENT already exists as a FILE (not a directory), so the real
    # mkdir(parents=True) inside _log_b4_compose_decision raises OSError — a genuine write
    # failure, not a mocked one.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    unwritable_log = blocker / "decisions.jsonl"

    direct_logged_ok = consolidate._log_b4_compose_decision(
        unwritable_log, fruit.cluster_id, [0, 1], fruit.action_items, composed,
    )
    assert direct_logged_ok is False
    assert not unwritable_log.exists()

    output, _ = _invoke_rejoin(
        fruit, _response(([0, 1], composed)), decision_log_path=unwritable_log,
    )

    assert output == fruit.action_items
    assert not unwritable_log.exists()


# B4 compose: bounded retry + rejection telemetry (A4, grouping-contract tighten). Not a new
# numbered guard — guard-6 is reserved for A3's supersession guard (A1 contract, config
# f552875 section 4). A schema/duplicate-invalid response gets ONE bounded, corrective retry
# before the whole cluster fails open. Recovers exactly the two
# failure shapes that caused all 3 whole-cluster fail-opens in the 2026-07-15 dogfood packet
# (dogfood-00/dogfood-08: duplicate membership; atlas-journal-038: malformed schema) — see
# config/design/recall-supersession-guard-detection-contract-2026-07-16.md's sibling A4 scope
# note. Retry does NOT apply to a well-formed response that a later per-group guard rejects
# (content-unsafe / temporal-conflict) — those are correct, deliberate degradations of valid
# model output, not malformed output worth re-asking for (pinned by the existing
# len(requests) == 1 assertions on those paths, unchanged below).


def test_bounded_retry_recovers_from_duplicate_membership_on_second_attempt(truth_fruit):
    calls = {"n": 0}
    valid_composed = (
        "The report uses LOW as a severity convention, and Recall stores append-versioned "
        "knowledge.jsonl revisions."
    )

    def completion(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response(
                ([0, 1], "first invalid composition"),
                ([1, 2], "second invalid composition"),
            )
        return _response(([2, 3], valid_composed))

    output, requests = _invoke_rejoin(truth_fruit, completion)

    assert len(requests) == 2
    assert _by_content(output, valid_composed)["content"] == valid_composed
    assert truth_fruit.action_items[0] in output
    assert truth_fruit.action_items[1] in output


def test_bounded_retry_recovers_from_malformed_schema_on_second_attempt(truth_fruit):
    calls = {"n": 0}
    valid_composed = (
        "The report uses LOW as a severity convention, and Recall stores append-versioned "
        "knowledge.jsonl revisions."
    )

    def completion(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"groups": [{"indices": "0,1", "content": "wrong type for indices"}]}
        return _response(([2, 3], valid_composed))

    output, requests = _invoke_rejoin(truth_fruit, completion)

    assert len(requests) == 2
    assert _by_content(output, valid_composed)["content"] == valid_composed


def test_bounded_retry_recovers_from_unparseable_response_on_second_attempt(truth_fruit):
    calls = {"n": 0}
    valid_composed = (
        "The report uses LOW as a severity convention, and Recall stores append-versioned "
        "knowledge.jsonl revisions."
    )

    def completion(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not-json"
        return _response(([2, 3], valid_composed))

    output, requests = _invoke_rejoin(truth_fruit, completion)

    assert len(requests) == 2
    assert _by_content(output, valid_composed)["content"] == valid_composed


def test_retry_is_bounded_and_still_fails_open_after_exhausting_attempts(truth_fruit, caplog):
    caplog.set_level(logging.WARNING)
    output, requests = _invoke_rejoin(
        truth_fruit,
        _response(([0, 1], "first invalid composition"), ([1, 2], "second invalid composition")),
    )

    assert output == truth_fruit.action_items
    assert len(requests) == consolidate._B4_COMPOSE_MAX_ATTEMPTS
    messages = _warning_messages(caplog)
    assert any("B4_COMPOSE_FAIL_OPEN" in message for message in messages)
    assert any(
        f"exhausted {consolidate._B4_COMPOSE_MAX_ATTEMPTS} attempt" in message
        for message in messages
    )


def test_retry_correction_prompt_names_the_specific_rejection_reason(truth_fruit):
    calls = {"n": 0}
    valid_composed = (
        "The report uses LOW as a severity convention, and Recall stores append-versioned "
        "knowledge.jsonl revisions."
    )

    def completion(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response(
                ([0, 1], "first invalid composition"),
                ([1, 2], "second invalid composition"),
            )
        return _response(([2, 3], valid_composed))

    _output, requests = _invoke_rejoin(truth_fruit, completion)

    assert "duplicate" in requests[1]["prompt"].lower()
    # attempt 0's prompt is byte-identical to the no-retry path — never mutated by the guard.
    assert requests[0]["prompt"] == requests[1]["prompt"].split("\n\nYour previous")[0]


# B4 compose rejection telemetry (A4): per-group rejections are loud and distinguishable by
# stage, closing the exact
# gap RESULTS.md named in the 2026-07-15 dogfood packet: "it does not record which per-group
# guard rejected each proposal... A content-versus-temporal breakdown therefore cannot be
# claimed from this run."


def test_invalid_index_group_rejection_is_logged_with_reason_and_indices(truth_fruit, caplog):
    caplog.set_level(logging.INFO)
    valid_composed = (
        "The report uses LOW as a severity convention, and Recall stores append-versioned "
        "knowledge.jsonl revisions."
    )
    _invoke_rejoin(
        truth_fruit,
        _response(
            ([0, 1, 999], "This composition was written for a fabricated membership set."),
            ([2, 3], valid_composed),
        ),
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "B4_GROUP_REJECTED" in message and "invalid-index" in message
        for message in messages
    )


def test_content_unsafe_group_rejection_is_logged_with_reason(truth_fruit, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    decision_path = tmp_path / "decisions.jsonl"
    _invoke_rejoin(
        truth_fruit, _response(([0, 1], "   ")), decision_log_path=decision_path,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "B4_GROUP_REJECTED" in message and "content-unsafe" in message
        for message in messages
    )
    entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
    rejections = [e for e in entries if e["action"] == "b4-group-rejected"]
    assert len(rejections) == 1
    assert rejections[0]["stage"] == "content-unsafe"
    assert rejections[0]["cluster_id"] == truth_fruit.cluster_id
    assert rejections[0]["member_indices"] == [0, 1]


def test_temporal_conflict_group_rejection_is_logged_with_reason(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    fruit = _real_pipeline([
        _FactSpec(
            "The 2025 signing key expires April 30.",
            temporal_role="expiry", resolved="2025-04-30",
        ),
        _FactSpec(
            "The 2026 signing key expires April 30.",
            temporal_role="expiry", resolved="2026-04-30",
        ),
    ], cluster_id="b4-temporal-telemetry")
    decision_path = tmp_path / "decisions.jsonl"
    _invoke_rejoin(
        fruit,
        _response(([0, 1], "Both signing keys expire April 30.")),
        decision_log_path=decision_path,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "B4_GROUP_REJECTED" in message and "temporal-conflict" in message
        for message in messages
    )
    entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
    rejections = [e for e in entries if e["action"] == "b4-group-rejected"]
    assert len(rejections) == 1
    assert rejections[0]["stage"] == "temporal-conflict"


def test_group_rejection_reasons_distinguish_content_from_temporal_in_one_decision_log(
    tmp_path,
):
    """The exact claim RESULTS.md said the 2026-07-15 packet could not make: a
    content-versus-temporal breakdown, queryable from the durable decision log rather than
    scraped from logger output."""
    fruit = _real_pipeline([
        _FactSpec(
            "The adapter failures reported in this cluster are unrelated to SQLite "
            "locking behavior observed during the retry-boundary investigation.",
        ),
        _FactSpec(
            "Recall stores durable node revisions in an append-versioned "
            "knowledge.jsonl file that survives process restarts without data loss.",
        ),
        _FactSpec(
            "The 2025 signing key expires April 30.",
            temporal_role="expiry", resolved="2025-04-30",
        ),
        _FactSpec(
            "The 2026 signing key expires April 30.",
            temporal_role="expiry", resolved="2026-04-30",
        ),
    ], cluster_id="b4-mixed-rejection-telemetry")
    decision_path = tmp_path / "decisions.jsonl"
    _invoke_rejoin(
        fruit,
        _response(
            ([0, 1], "   "),
            ([2, 3], "Both signing keys expire April 30."),
        ),
        decision_log_path=decision_path,
    )

    entries = [json.loads(line) for line in decision_path.read_text().splitlines() if line]
    rejections = {e["stage"]: e for e in entries if e["action"] == "b4-group-rejected"}
    assert set(rejections) == {"content-unsafe", "temporal-conflict"}
    assert rejections["content-unsafe"]["member_indices"] == [0, 1]
    assert rejections["temporal-conflict"]["member_indices"] == [2, 3]
