from pathlib import Path

from medhorizon_videorag.graph_rag import (
    EvidenceInterval,
    GraphNode,
    VideoEvidenceGraph,
)
from medhorizon_videorag.graph_rag.qa_experiment import (
    OpenAICompatibleGraphQA,
    build_event_observation_catalog,
    select_event_frame_groups,
    strip_explicit_time_range,
)


def _graph(tmp_path: Path) -> VideoEvidenceGraph:
    frame_paths = []
    for index in range(8):
        path = tmp_path / f"{index:02d}.jpg"
        path.write_bytes(b"frame")
        frame_paths.append(str(path))
    interval = EvidenceInterval(
        "087",
        0.0,
        64.0,
        frame_paths,
        metadata={"clip_id": "087_vgent_00000", "clip_index": 0},
    )
    clip = GraphNode(
        "clip:087_vgent_00000",
        "087",
        "segment",
        "A needle-like instrument contacts reddish tissue.",
        [interval],
        metadata={
            "clip_id": "087_vgent_00000",
            "observation": {
                "observed_facts": {
                    "visible_instruments": ["needle-like instrument"],
                    "actions": [
                        {
                            "subject": "needle-like instrument",
                            "action": "contacts",
                            "target": "reddish tissue",
                        }
                    ],
                    "state_changes": [],
                },
                "medical_inferences": [
                    {"inference": "A phase label that must not enter the catalog."}
                ],
            },
        },
    )
    event = GraphNode(
        "event:087:00000",
        "087",
        "temporal_event",
        "contact | needle_like_instrument, tissue",
        [EvidenceInterval("087", 0.0, 64.0)],
        metadata={
            "supporting_clip_ids": ["087_vgent_00000"],
            "predicates": ["contact"],
            "concepts": ["needle_like_instrument", "tissue"],
            "representative_evidence": [{"clip_id": "087_vgent_00000"}],
        },
    )
    return VideoEvidenceGraph("087", [clip, event], [])


def test_strip_explicit_time_range_keeps_the_question_semantics() -> None:
    assert (
        strip_explicit_time_range(
            "Review the operative activity from 12:44 to 13:44; which phase does it correspond to?"
        )
        == "Review the operative activity; which phase does it correspond to?"
    )


def test_catalog_uses_observed_facts_but_not_medical_inferences(
    tmp_path: Path,
) -> None:
    catalog = build_event_observation_catalog(_graph(tmp_path))

    assert catalog[0]["event_id"] == "event:087:00000"
    assert catalog[0]["observations"][0]["visible_instruments"] == [
        "needle-like instrument"
    ]
    assert catalog[0]["concepts"] == ["needle_like_instrument"]
    assert "phase label" not in str(catalog).lower()


def test_onset_selection_returns_uniform_traceable_frames(tmp_path: Path) -> None:
    groups = select_event_frame_groups(
        _graph(tmp_path),
        ["event:087:00000"],
        frames_per_event=4,
        prefer_onset=True,
    )

    assert groups[0]["clip_id"] == "087_vgent_00000"
    assert groups[0]["selection"] == "event_onset_clip"
    assert len(groups[0]["reader_frame_paths"]) == 4
    assert groups[0]["reader_frame_paths"][0].endswith("00.jpg")
    assert groups[0]["reader_frame_paths"][-1].endswith("07.jpg")


def test_query_conditioned_activity_rerank_and_verification() -> None:
    client = OpenAICompatibleGraphQA.__new__(OpenAICompatibleGraphQA)
    client._plain_text_response = lambda prompt, max_tokens: (  # type: ignore[method-assign]
        "open_activity:00002, open_activity:00001"
    )
    catalog = [
        {"segment_id": "open_activity:00001", "activity_label": "first"},
        {"segment_id": "open_activity:00002", "activity_label": "second"},
    ]

    segment_ids, rationale = client.rerank_activity_segments(
        "Target Phase", catalog, top_segments=2
    )

    assert segment_ids == ["open_activity:00002", "open_activity:00001"]
    assert rationale == ""
    client._vision_json = lambda content, max_tokens: {  # type: ignore[method-assign]
        "selected_segment_id": "open_activity:00002",
        "confidence": "medium",
        "rationale": "frames support the candidate",
    }
    verification = client.verify_phase_activity_candidates("Target Phase", catalog, [])
    assert verification == {
        "selected_segment_id": "open_activity:00002",
        "confidence": "medium",
        "rationale": "frames support the candidate",
    }
