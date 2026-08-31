import json
from copy import deepcopy

import pytest

from medhorizon_videorag.graph_rag import (
    build_evidence_graph,
    normalize_description_rows,
)
from medhorizon_videorag.graph_rag.mention_binding import bind_mention


@pytest.mark.parametrize(
    "query,candidates,expected,method",
    [
        (
            "metal instrument with clamping end",
            ["metal instrument with pointed tip", "metal instrument with clamping end"],
            1,
            "exact_surface",
        ),
        (
            "grasper",
            [
                "metal needle holder with serrated jaws",
                "metal grasper with textured tips",
            ],
            1,
            "unique_head",
        ),
        (
            "tubular structure",
            ["reddish tissue with surface vessels", "smooth tubular structure"],
            1,
            "head_attributes",
        ),
        (
            "metallic tool with flat tip",
            ["metal tool with curved tip", "metal tool with flat jaws"],
            1,
            "head_attributes",
        ),
        (
            "blue thread",
            ["white thread-like material", "thin blue thread-like material"],
            1,
            "head_attributes",
        ),
        (
            "needle-like instrument",
            [
                "metal needle-like instrument holding blue thread",
                "blue thread-like material",
            ],
            0,
            "unique_head",
        ),
        ("tool", ["metal forceps"], 0, "unique_head"),
    ],
)
def test_binding_uses_reference_evidence_not_concept_or_order(
    query, candidates, expected, method
):
    pairs = [(str(i), text) for i, text in enumerate(candidates)]
    for ordered in [pairs, pairs[::-1]]:
        chosen, info = bind_mention(query, ordered)
        assert chosen == str(expected)
        assert info["method"] == method
        assert info["status"] == "resolved"
        assert info["physical_identity_confirmed"] is False


@pytest.mark.parametrize(
    "query,candidates,status",
    [
        (
            "instrument",
            ["metal tool with flat tip", "metal tool with curved tip"],
            "ambiguous",
        ),
        ("metal forceps", ["metal forceps", "metal forceps"], "ambiguous"),
        ("blue instrument", ["white instrument", "gray instrument"], "unmatched"),
        ("curved instrument", ["metal instrument"], "unmatched"),
        ("needle holder", ["metal grasper"], "unmatched"),
        ("unidentified object", ["metal instrument"], "unmatched"),
        ("clip", ["mesh"], "unmatched"),
        ("serrated instrument", ["non-serrated instrument"], "unmatched"),
        ("needle holder", ["not a needle holder"], "unmatched"),
        ("needle holder", ["metal needle holder or forceps"], "unmatched"),
    ],
)
def test_binding_abstains_without_unique_supported_reference(query, candidates, status):
    pairs = [(str(i), text) for i, text in enumerate(candidates)]
    chosen, info = bind_mention(query, pairs)
    assert chosen is None
    assert info["status"] == status
    assert info["selected_mention_id"] is None


def _row(index=0):
    return {
        "clip_id": f"case_{index}",
        "video_id": "case",
        "clip_index": index,
        "start_seconds": index * 64,
        "end_seconds": (index + 1) * 64,
        "description": {
            "summary": "Two tools hold tissue.",
            "observed_facts": {
                "visible_instruments": ["metal forceps", "blue forceps"],
                "visible_anatomy": ["tissue"],
                "visible_objects": [],
                "actions": [
                    {"subject": "forceps", "action": "holds", "target": "tissue"}
                ]
                * 2,
                "state_changes": [],
                "visual_evidence": [],
            },
            "uncertainties": [],
            "medical_inferences": [],
        },
    }


def test_ambiguous_arguments_are_independent_traceable_and_not_continuity_support():
    rows = [_row(0), _row(1)]
    before = deepcopy(rows)
    artifacts = build_evidence_graph(rows)
    assert rows == before
    nodes = {n.id: n for n in artifacts.graph.nodes}
    for clip in artifacts.normalized_clips:
        left, right = clip.actions
        assert left.subject_mention_id != right.subject_mention_id
        for action in clip.actions:
            binding = action.subject_binding
            assert binding["status"] == "ambiguous"
            assert len(binding["compatible_mention_ids"]) == 2
            mention = nodes[action.subject_mention_id]
            assert mention.label == "forceps"
            assert mention.metadata["source_field"] == "action_subject"
            assert mention.metadata["argument_binding"] == binding
    for event in artifacts.temporal_events:
        for detail in event.merge_details:
            assert not any(
                role["role"] == "subject" for role in detail["shared_action_roles"]
            )
    assert not any(
        e.relation == "possible_continuation" and e.source.startswith("action:")
        for e in artifacts.graph.edges
    )
    assert artifacts.report["argument_binding_status_counts"]["ambiguous"] == 4


def test_exact_binding_can_cross_old_canonical_error_without_rewriting_mentions():
    row = _row()
    facts = row["description"]["observed_facts"]
    facts["visible_instruments"] = [
        "metal instrument",
        "needle-like instrument holding thread-like material",
    ]
    facts["actions"] = [
        {
            "subject": facts["visible_instruments"][1],
            "action": "pulls",
            "target": "tissue",
        }
    ]
    clip = normalize_description_rows([row])[0]
    action = clip.actions[0]
    linked = next(m for m in clip.mentions if m.id == action.subject_mention_id)
    assert linked.surface == facts["visible_instruments"][1]
    assert action.subject_binding["method"] == "exact_surface"
    # The P0 change is reference binding, not the separate entity ontology fix.
    assert linked.canonical == "thread_like_material"


def test_acceptance_audit_detects_mislink_and_does_not_reward_abstaining_on_exact():
    from experiments.evaluate_mention_binding import audit_bindings

    row = _row()
    row["description"]["observed_facts"]["actions"] = [
        {"subject": "blue forceps", "action": "holds", "target": "tissue"}
    ]
    good = build_evidence_graph([row]).graph.to_dict()
    bad = deepcopy(good)
    wrong = next(
        n["id"]
        for n in bad["nodes"]
        if n["node_type"] == "entity_mention" and n["label"] == "metal forceps"
    )
    edge = next(e for e in bad["edges"] if e["relation"] == "has_subject")
    edge["target"] = wrong
    fixed, _ = audit_bindings(bad, good)
    assert fixed["passed"]
    assert fixed["counts"]["baseline_mislinks_now_resolved_exact"] == 1
    failed, _ = audit_bindings(good, bad)
    assert not failed["passed"]
    assert failed["counts"]["v3_exact_alternative_mislinks"] == 1
    action = next(n for n in good["nodes"] if n["node_type"] == "action_event")
    action["metadata"]["subject_binding"]["status"] = "unmatched"
    failed, _ = audit_bindings(bad, good)
    assert not failed["passed"]
    assert failed["counts"]["unique_exact_not_resolved"] == 1


def test_rebuild_cli_preserves_baseline_and_refuses_existing_output(
    tmp_path, monkeypatch
):
    from experiments.evaluate_mention_binding import main

    row = _row()
    old = tmp_path / "baseline.json"
    old.write_text(json.dumps(build_evidence_graph([row]).graph.to_dict()))
    original = old.read_bytes()
    descriptions = tmp_path / "descriptions.jsonl"
    descriptions.write_text(json.dumps(row) + "\n")
    output = tmp_path / "v3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_mention_binding",
            "--baseline-graph",
            str(old),
            "--descriptions",
            str(descriptions),
            "--output-dir",
            str(output),
        ],
    )
    main()
    assert old.read_bytes() == original
    assert json.loads((output / "mention_binding_audit/report.json").read_text())[
        "passed"
    ]
    graph_bytes = (output / "evidence_graph.json").read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert (output / "evidence_graph.json").read_bytes() == graph_bytes
