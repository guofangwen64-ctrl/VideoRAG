from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from medhorizon_videorag.vgent_baseline import (
    MedicalStreamingExtractor,
    VgentSlicingConfig,
    load_video_plan,
    safe_video_key,
    save_video_plan,
    video_plan_cache_complete,
)


class _FakeCapture:
    def __init__(
        self, total_frames: int, fps: float, readable_frames: int | None = None
    ) -> None:
        self.total_frames = total_frames
        self.fps = fps
        self.readable_frames = (
            total_frames if readable_frames is None else readable_frames
        )
        self.position = 0

    def isOpened(self) -> bool:
        return True

    def get(self, field: int) -> float:
        return self.fps if field == 1 else float(self.total_frames)

    def read(self):
        if self.position >= self.readable_frames:
            return False, None
        frame = self.position
        self.position += 1
        return True, frame

    def release(self) -> None:
        pass


def test_medical_streaming_extracts_one_pass_clip_cache(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    def imwrite(path: str, frame: int, options: list[int]) -> bool:
        Path(path).write_text(str(frame), encoding="utf-8")
        return options == [3, 95]

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        IMWRITE_JPEG_QUALITY=3,
        VideoCapture=lambda _: _FakeCapture(130, 1.0),
        imwrite=imwrite,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    extractor = MedicalStreamingExtractor(
        VgentSlicingConfig(min_sampled_frames=2),
        ffmpeg_fallback_min_incomplete_ratio=None,
    )

    plan = extractor.extract("case/1", source, tmp_path / "frames")

    assert plan.sampled_frames == 130
    assert [len(clip.frame_paths) for clip in plan.clips] == [64, 64, 2]
    assert plan.metadata["decoder"] == "opencv"
    assert plan.clips[0].end_seconds == 64.0
    assert safe_video_key("case/1") in plan.clips[0].frame_paths[0]
    assert video_plan_cache_complete(plan) is True


def test_per_video_manifest_round_trip_is_atomic(tmp_path: Path) -> None:
    from medhorizon_videorag.vgent_baseline import VgentSlicingPlanner

    plan = VgentSlicingPlanner().plan("case-1", "case-1.mp4", 65.0)
    destination = tmp_path / "case.json"

    save_video_plan(plan, destination)
    restored = load_video_plan(destination)

    assert restored == plan
    assert not (tmp_path / ".case.json.tmp").exists()
    assert video_plan_cache_complete(restored) is False


def test_medical_streaming_ffmpeg_fallback_preserves_clip_grouping(
    tmp_path: Path, monkeypatch
) -> None:
    import medhorizon_videorag.vgent_baseline.streaming as streaming_module

    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        IMWRITE_JPEG_QUALITY=3,
        VideoCapture=lambda _: _FakeCapture(65, 1.0, readable_frames=0),
        imwrite=lambda *_: False,
    )

    def fake_run(arguments: list[str], **_: object) -> SimpleNamespace:
        pattern = Path(arguments[-1])
        for index in range(65):
            destination = pattern.parent / f"{index + 1:08d}.jpg"
            destination.write_bytes(b"frame")
        return SimpleNamespace(returncode=0)

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(streaming_module.subprocess, "run", fake_run)
    extractor = MedicalStreamingExtractor(
        VgentSlicingConfig(frames_per_clip=64),
        ffmpeg_fallback_min_incomplete_ratio=0.0,
    )

    plan = extractor.extract("case-ffmpeg", source, tmp_path / "frames")

    assert plan.metadata["decoder"] == "ffmpeg"
    assert [len(clip.frame_paths) for clip in plan.clips] == [64, 1]
    assert video_plan_cache_complete(plan) is True
