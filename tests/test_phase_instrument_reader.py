from types import MethodType

from medhorizon_videorag.graph_rag import OpenAICompatibleGraphQA


def test_option_verifier_returns_assessments_and_filters_track_ids() -> None:
    reader = object.__new__(OpenAICompatibleGraphQA)
    reader.max_image_pixels = 1
    captured = {}

    def fake_vision_json(self, content, *, max_tokens):
        captured["prompt"] = content[0]["text"]
        captured["max_tokens"] = max_tokens
        return {
            "choice": "B",
            "selected_track_ids": ["track:kept", "track:unknown"],
            "rationale": "option B has visible support",
            "option_assessments": [
                {
                    "option_label": "A",
                    "support": [],
                    "contradiction": ["not tubular"],
                    "missing_evidence": ["no suction tip"],
                    "matched_track_ids": [],
                    "confidence": "low",
                },
                {
                    "option_label": "B",
                    "support": ["jawed instrument holds material"],
                    "contradiction": [],
                    "missing_evidence": [],
                    "matched_track_ids": ["track:kept"],
                    "confidence": "medium",
                },
            ],
        }

    reader._vision_json = MethodType(fake_vision_json, reader)
    reader_input = {
        "phase_label": "Example Phase",
        "candidate_tracks": [
            {
                "track_id": "track:kept",
                "graph_rank": 1,
                "label": "grasping instrument",
                "appearance_family": "grasping_instrument",
                "appearance_signature": {},
                "surface_forms": ["metal forceps"],
                "action_roles": ["subject:hold"],
                "reader_clip_ids": ["clip:1"],
                "option_matches": [{"option_label": "B", "score": 0.8}],
            }
        ],
        "evidence_groups": [],
    }

    choice, rationale, track_ids, assessments = (
        reader.answer_phase_instrument_with_option_verifier(
            "At phase onset, what instrument is visible?",
            ["A. Aspirator", "B. Forceps"],
            reader_input,
        )
    )

    assert choice == "B"
    assert "visible support" in rationale
    assert track_ids == ["track:kept"]
    assert assessments[1]["option_label"] == "B"
    assert "option_assessments" in captured["prompt"]
    assert captured["max_tokens"] == 900
