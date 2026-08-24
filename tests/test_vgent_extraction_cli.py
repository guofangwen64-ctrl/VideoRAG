from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.extract_vgent_streaming import (
    _remove_resolved_errors,
    _select_videos,
)


def test_selects_exact_video_keys_in_requested_order() -> None:
    videos = [SimpleNamespace(key="047"), SimpleNamespace(key="079")]

    selected = _select_videos(videos, "079,047", None)

    assert [item.key for item in selected] == ["079", "047"]


def test_video_key_selection_rejects_unknown_or_conflicting_limit() -> None:
    videos = [SimpleNamespace(key="079")]

    with pytest.raises(ValueError, match="Unknown video keys"):
        _select_videos(videos, "missing", None)
    with pytest.raises(ValueError, match="cannot be used together"):
        _select_videos(videos, "079", 1)


def test_resolved_extraction_errors_are_removed(tmp_path: Path) -> None:
    errors = tmp_path / "errors.jsonl"
    errors.write_text(
        '{"video_key":"079","error":"old"}\n{"video_key":"047","error":"retry"}\n',
        encoding="utf-8",
    )

    _remove_resolved_errors(errors, {"079"})

    assert errors.read_text(encoding="utf-8") == (
        '{"video_key": "047", "error": "retry"}\n'
    )
    _remove_resolved_errors(errors, {"047"})
    assert not errors.exists()
