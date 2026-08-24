from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.describe_vgent_clips import (
    _load_segment_cache,
    _parse_clip_indices,
    _remove_resolved_errors,
    _remove_resolved_segment_cache,
)
from medhorizon_videorag.vgent_baseline.description import (
    DESCRIPTION_PROMPT,
    DESCRIPTION_PROMPT_VERSION,
    OBSERVATION_FIRST_SYSTEM_PROMPT,
    OpenAICompatibleClipDescriber,
    find_summary_rule_violations,
    merge_segment_descriptions,
    prepare_request_frame_paths,
    select_even_full_clips,
    select_full_clips_by_index,
    split_clip_frame_batches,
)
from medhorizon_videorag.vgent_baseline.schemas import VgentClipPlan


def _clip(tmp_path: Path, index: int, frame_count: int = 64) -> VgentClipPlan:
    frame_paths = []
    for frame_index in range(frame_count):
        path = tmp_path / f"{index}_{frame_index}.jpg"
        path.touch()
        frame_paths.append(str(path))
    return VgentClipPlan(
        id=f"v_vgent_{index:05d}",
        video_id="v",
        video_path="v.mp4",
        clip_index=index,
        start_seconds=float(index * 64),
        end_seconds=float(index * 64 + frame_count),
        sample_start_index=index * 64,
        sample_end_index=index * 64 + frame_count,
        sampled_frame_count=frame_count,
        effective_fps=1.0,
        is_partial=frame_count < 64,
        frame_paths=frame_paths,
    )


def test_selects_ten_even_complete_clips_and_excludes_partial(tmp_path: Path) -> None:
    clips = [_clip(tmp_path, index) for index in range(87)]
    clips.append(_clip(tmp_path, 87, 60))

    selected = select_even_full_clips(clips, 10, frames_per_request=64)

    assert [clip.clip_index for clip in selected] == [
        0,
        10,
        19,
        29,
        38,
        48,
        57,
        67,
        76,
        86,
    ]
    assert all(len(clip.frame_paths) == 64 for clip in selected)


def test_rejects_insufficient_complete_clips(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only 1 are available"):
        select_even_full_clips([_clip(tmp_path, 0)], 2, frames_per_request=64)


def test_selects_exact_complete_clip_indices(tmp_path: Path) -> None:
    clips = [_clip(tmp_path, index) for index in range(4)]

    selected = select_full_clips_by_index(clips, [3, 0, 2])

    assert [clip.clip_index for clip in selected] == [3, 0, 2]


def test_rejects_partial_or_duplicate_explicit_clip_indices(tmp_path: Path) -> None:
    clips = [_clip(tmp_path, 0), _clip(tmp_path, 1, 60)]

    with pytest.raises(ValueError, match="must be unique"):
        select_full_clips_by_index(clips, [0, 0])
    with pytest.raises(ValueError, match="not a complete"):
        select_full_clips_by_index(clips, [1])


def test_pads_partial_tail_to_exact_request_length(tmp_path: Path) -> None:
    clip = _clip(tmp_path, 87, 60)

    request_paths = prepare_request_frame_paths(clip, frames_per_request=64)

    assert len(request_paths) == 64
    assert request_paths[:60] == clip.frame_paths
    assert request_paths[60:] == [clip.frame_paths[-1]] * 4


def test_splits_parent_clip_into_two_contiguous_32_frame_requests(
    tmp_path: Path,
) -> None:
    clip = _clip(tmp_path, 2)

    batches = split_clip_frame_batches(clip)

    assert len(batches) == 2
    assert batches[0].frame_paths == clip.frame_paths[:32]
    assert batches[1].frame_paths == clip.frame_paths[32:]
    assert [(item.start_seconds, item.end_seconds) for item in batches] == [
        (128.0, 160.0),
        (160.0, 192.0),
    ]
    assert all(item.padding_frame_count == 0 for item in batches)


def test_splits_parent_clip_into_four_contiguous_16_frame_requests(
    tmp_path: Path,
) -> None:
    clip = _clip(tmp_path, 2)

    batches = split_clip_frame_batches(clip, frames_per_request=16)

    assert len(batches) == 4
    assert [item.frame_paths for item in batches] == [
        clip.frame_paths[0:16],
        clip.frame_paths[16:32],
        clip.frame_paths[32:48],
        clip.frame_paths[48:64],
    ]
    assert [(item.start_seconds, item.end_seconds) for item in batches] == [
        (128.0, 144.0),
        (144.0, 160.0),
        (160.0, 176.0),
        (176.0, 192.0),
    ]
    assert all(item.padding_frame_count == 0 for item in batches)


def test_splits_and_pads_each_half_of_partial_tail(tmp_path: Path) -> None:
    clip = _clip(tmp_path, 2, 27)

    batches = split_clip_frame_batches(clip)

    assert [item.source_frame_count for item in batches] == [14, 13]
    assert [item.padding_frame_count for item in batches] == [18, 19]
    assert all(len(item.frame_paths) == 32 for item in batches)
    assert batches[0].frame_paths[-1] == clip.frame_paths[13]
    assert batches[1].frame_paths[0] == clip.frame_paths[14]
    assert batches[1].frame_paths[-1] == clip.frame_paths[-1]


def test_merges_segment_descriptions_without_model_inference() -> None:
    first = {
        "summary": "A metal instrument contacts reddish tissue.",
        "observed_facts": {
            "visible_anatomy": ["reddish tissue"],
            "visible_instruments": ["metal instrument"],
            "visible_objects": [],
            "actions": [
                {
                    "subject": "metal instrument",
                    "action": "contacts",
                    "target": "reddish tissue",
                }
            ],
            "state_changes": [],
            "visual_evidence": ["The instrument tip touches the tissue."],
        },
        "medical_inferences": [],
        "uncertainties": [],
    }
    second = {
        **first,
        "summary": "The metal instrument moves away from reddish tissue.",
        "observed_facts": {
            **first["observed_facts"],
            "actions": [
                {
                    "subject": "metal instrument",
                    "action": "moves away from",
                    "target": "reddish tissue",
                }
            ],
        },
    }

    merged = merge_segment_descriptions([first, second])

    assert merged["summary"] == f"{first['summary']} {second['summary']}"
    assert merged["observed_facts"]["visible_anatomy"] == ["reddish tissue"]
    assert len(merged["observed_facts"]["actions"]) == 2
    assert merged["medical_inferences"] == []


def test_observation_first_prompt_contract() -> None:
    assert DESCRIPTION_PROMPT_VERSION == "medical_clip_observation_first_v10"
    assert "The summary MUST contain only directly visible information." in (
        DESCRIPTION_PROMPT
    )
    assert '"red fluid" instead of "active bleeding"' in DESCRIPTION_PROMPT
    assert "Do not use external context, video title, dataset metadata" in (
        DESCRIPTION_PROMPT
    )
    assert "return an empty medical_inferences array" in DESCRIPTION_PROMPT
    assert "The action may represent suturing." not in DESCRIPTION_PROMPT
    assert "literal visual transcription system" in OBSERVATION_FIRST_SYSTEM_PROMPT
    assert "silently check the" in OBSERVATION_FIRST_SYSTEM_PROMPT
    assert "exactly one short sentence of at most 30 words" in (
        OBSERVATION_FIRST_SYSTEM_PROMPT
    )


def test_finds_summary_rule_violations_without_substring_false_positives() -> None:
    summary = "Red fluid is visible in a surgical field during tissue repair."

    assert find_summary_rule_violations(summary) == [
        "surgical",
        "surgical field",
        "repair",
    ]
    assert (
        find_summary_rule_violations("A small tubular structure remains visible.") == []
    )


def test_resolved_description_errors_are_removed(tmp_path: Path) -> None:
    errors = tmp_path / "errors.jsonl"
    errors.write_text(
        '{"clip_id":"done","error":"old"}\n{"clip_id":"pending","error":"retry"}\n',
        encoding="utf-8",
    )

    _remove_resolved_errors(errors, {"done"})

    assert errors.read_text(encoding="utf-8") == (
        '{"clip_id": "pending", "error": "retry"}\n'
    )
    _remove_resolved_errors(errors, {"pending"})
    assert not errors.exists()


def test_segment_cache_keeps_only_unresolved_parent_clips(tmp_path: Path) -> None:
    cache = tmp_path / ".descriptions.segments.jsonl"
    cache.write_text(
        '{"segment_id":"done:segment:00","clip_id":"done"}\n'
        '{"segment_id":"pending:segment:00","clip_id":"pending"}\n',
        encoding="utf-8",
    )

    assert set(_load_segment_cache(cache)) == {
        "done:segment:00",
        "pending:segment:00",
    }
    _remove_resolved_segment_cache(cache, {"done"})

    assert set(_load_segment_cache(cache)) == {"pending:segment:00"}
    _remove_resolved_segment_cache(cache, {"pending"})
    assert not cache.exists()


def test_parses_unique_non_negative_clip_indices() -> None:
    assert _parse_clip_indices("32, 7", option="--skip-clip-indices") == [32, 7]
    assert _parse_clip_indices(None, option="--skip-clip-indices") == []
    with pytest.raises(ValueError, match="unique, non-negative"):
        _parse_clip_indices("32,32", option="--skip-clip-indices")
    with pytest.raises(ValueError, match="unique, non-negative"):
        _parse_clip_indices("-1", option="--skip-clip-indices")


def test_retries_transient_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class ServerError(Exception):
        status_code = 500
        response = SimpleNamespace(headers={})

    completions = SimpleNamespace()
    completions.calls = 0

    def create(**request):
        assert request["model"] == "test-model"
        completions.calls += 1
        if completions.calls == 1:
            raise ServerError("server busy")
        message = SimpleNamespace(content='{"summary":"visible activity"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions.create = create
    describer = OpenAICompatibleClipDescriber.__new__(OpenAICompatibleClipDescriber)
    describer.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    describer.model = "test-model"
    describer.max_tokens = 128
    describer.response_format_json = False
    describer.request_extra_body = {}
    describer.max_retries = 2
    describer.initial_retry_seconds = 30
    describer.max_retry_seconds = 300
    describer.non_retryable_status_codes = frozenset()
    describer.last_attempt_count = 0
    sleeps = []
    monkeypatch.setattr(
        "medhorizon_videorag.vgent_baseline.description.time.sleep",
        sleeps.append,
    )

    payload = describer._request_payload([{"role": "user", "content": "test"}])

    assert payload == {"summary": "visible activity"}
    assert completions.calls == 2
    assert describer.last_attempt_count == 2
    assert sleeps == [30]


def test_does_not_retry_configured_http_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ServerError(Exception):
        status_code = 500
        response = SimpleNamespace(headers={})

    completions = SimpleNamespace(calls=0)

    def create(**request):
        assert request["model"] == "test-model"
        completions.calls += 1
        raise ServerError("server busy")

    completions.create = create
    describer = OpenAICompatibleClipDescriber.__new__(OpenAICompatibleClipDescriber)
    describer.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    describer.model = "test-model"
    describer.max_tokens = 128
    describer.response_format_json = False
    describer.request_extra_body = {}
    describer.max_retries = 8
    describer.initial_retry_seconds = 30
    describer.max_retry_seconds = 300
    describer.non_retryable_status_codes = frozenset({500})
    describer.last_attempt_count = 0
    sleeps = []
    monkeypatch.setattr(
        "medhorizon_videorag.vgent_baseline.description.time.sleep",
        sleeps.append,
    )

    with pytest.raises(ServerError, match="server busy"):
        describer._request_payload([{"role": "user", "content": "test"}])

    assert completions.calls == 1
    assert describer.last_attempt_count == 1
    assert sleeps == []
