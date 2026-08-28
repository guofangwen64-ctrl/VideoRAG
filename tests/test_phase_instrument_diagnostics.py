from experiments.analyze_phase_instrument_diagnostics import (
    _max_option_overlap,
    _option_by_label,
    _overlaps_any,
    _score_margin,
)


def test_option_by_label_accepts_plain_letter_labels() -> None:
    options = ["A. Bipolar forceps", "B. Needle holder"]

    assert _option_by_label("b", options) == "B. Needle holder"


def test_max_option_overlap_uses_option_text_against_track_terms() -> None:
    options = ["A. Bipolar forceps", "B. Needle holder"]
    tracks = [
        {"track_id": "t1", "label": "bipolar tool", "surface_forms": ["forceps"]},
        {"track_id": "t2", "label": "needle holder", "surface_forms": []},
    ]

    overlap = _max_option_overlap("B", options, tracks)

    assert overlap["option"] == "B. Needle holder"
    assert overlap["track_id"] == "t2"
    assert overlap["matched_tokens"] == ["needle"]


def test_overlaps_any_returns_none_when_no_anchor_exists() -> None:
    segment = {"start_seconds": 10, "end_seconds": 20}

    assert _overlaps_any(segment, []) is None


def test_overlaps_any_detects_half_open_interval_overlap() -> None:
    segment = {"start_seconds": 10, "end_seconds": 20}
    windows = [{"start_seconds": 19, "end_seconds": 30}]

    assert _overlaps_any(segment, windows) is True


def test_score_margin_uses_top_two_retrieval_candidates() -> None:
    fallback = {
        "retrieval_candidates": [
            {"retrieval_score": 5.25},
            {"retrieval_score": 4.0},
        ]
    }

    assert _score_margin(fallback) == 1.25
