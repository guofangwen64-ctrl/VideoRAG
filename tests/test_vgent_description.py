from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.describe_vgent_clips import _remove_resolved_errors
from medhorizon_videorag.vgent_baseline.description import (
    DESCRIPTION_PROMPT,
    DESCRIPTION_PROMPT_VERSION,
    OBSERVATION_FIRST_SYSTEM_PROMPT,
    OpenAICompatibleClipDescriber,
    find_summary_rule_violations,
    prepare_request_frame_paths,
    select_even_full_clips,
    select_full_clips_by_index,
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
