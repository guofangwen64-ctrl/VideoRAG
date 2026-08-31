from experiments.analyze_phase_candidate_generation import (
    _canonical,
    _intervals_overlap,
    _nearest_events_for_phase,
    _sample_frame_paths,
    _summarize,
)


def test_canonical_ignores_case_and_punctuation() -> None:
    assert _canonical("Right Pelvic and Iliac Lymph Node Dissection") == (
        "rightpelvicandiliaclymphnodedissection"
    )


def test_intervals_overlap_uses_half_open_ranges() -> None:
    assert _intervals_overlap((0.0, 10.0), (9.0, 12.0)) is True
    assert _intervals_overlap((0.0, 10.0), (10.0, 12.0)) is False


def test_sample_frame_paths_keeps_first_middle_last() -> None:
    node = {"evidence": [{"frame_paths": [f"f{i}.jpg" for i in range(5)]}]}

    assert _sample_frame_paths(node) == ["f0.jpg", "f2.jpg", "f4.jpg"]


def test_nearest_events_for_phase_uses_graph_tokens() -> None:
    index = {
        "temporal_events": [
            {
                "id": "event:1",
                "label": "generic tissue manipulation",
                "evidence": [{"start_seconds": 0, "end_seconds": 10}],
                "metadata": {"supporting_clip_ids": ["c0"], "concepts": []},
            },
            {
                "id": "event:2",
                "label": "iliac vessel exposure",
                "evidence": [{"start_seconds": 10, "end_seconds": 20}],
                "metadata": {"supporting_clip_ids": ["c1"], "concepts": ["lymph"]},
            },
        ],
        "action_events": [],
        "entity_mentions": [],
    }

    events = _nearest_events_for_phase(
        index, "Right Pelvic and Iliac Lymph Node Dissection"
    )

    assert [event["id"] for event in events] == ["event:2"]


def test_summarize_counts_topk_and_top1_gap() -> None:
    rows = [
        {
            "gt_phase": "A",
            "candidate_generation": {
                "topk_hit": True,
                "top1_hit": False,
                "best_rank": 2,
            },
            "temporal_evidence": {"windows": [{"start_seconds": 0, "end_seconds": 1}]},
        },
        {
            "gt_phase": "B",
            "candidate_generation": {
                "topk_hit": False,
                "top1_hit": False,
                "best_rank": None,
            },
            "temporal_evidence": {"windows": []},
        },
    ]

    summary = _summarize(rows)

    assert summary["candidate_topk_recall"] == 0.5
    assert summary["topk_not_top1"] == 1
    assert summary["missing_without_temporal_anchor"] == 1
