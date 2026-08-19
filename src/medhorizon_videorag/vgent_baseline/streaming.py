from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .schemas import VgentVideoPlan
from .slicing import VgentSlicingConfig, VgentSlicingPlanner


class MedicalStreamingExtractor:
    """Decode once, sample at a fixed rate, and persist one clip at a time."""

    def __init__(
        self,
        config: VgentSlicingConfig | None = None,
        *,
        opencv_jpeg_quality: int = 95,
        ffmpeg_jpeg_quality: int = 2,
        ffmpeg_fallback_min_incomplete_ratio: float | None = 0.01,
    ) -> None:
        self.config = config or VgentSlicingConfig(mode="medical_streaming")
        if self.config.mode != "medical_streaming":
            raise ValueError(
                "MedicalStreamingExtractor requires mode=medical_streaming"
            )
        if not 0 <= opencv_jpeg_quality <= 100:
            raise ValueError("opencv_jpeg_quality must be between 0 and 100")
        if not 2 <= ffmpeg_jpeg_quality <= 31:
            raise ValueError("ffmpeg_jpeg_quality must be between 2 and 31")
        if (
            ffmpeg_fallback_min_incomplete_ratio is not None
            and not 0 <= ffmpeg_fallback_min_incomplete_ratio <= 1
        ):
            raise ValueError("ffmpeg fallback ratio must be between 0 and 1")
        self.opencv_jpeg_quality = opencv_jpeg_quality
        self.ffmpeg_jpeg_quality = ffmpeg_jpeg_quality
        self.ffmpeg_fallback_min_incomplete_ratio = ffmpeg_fallback_min_incomplete_ratio

    def extract(
        self,
        video_id: str,
        video_path: str | Path,
        frame_root: str | Path,
        *,
        annotation_duration_seconds: float | None = None,
    ) -> VgentVideoPlan:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Install video dependencies: pip install -e '.[video]'"
            ) from error

        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError(f"Video does not exist: {source}")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Cannot open video: {source}")

        try:
            native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            total_native_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if native_fps <= 0 or total_native_frames <= 0:
                raise ValueError(f"Invalid video metadata: {source}")
            duration_seconds = total_native_frames / native_fps
            plan = VgentSlicingPlanner(self.config).plan(
                video_id, str(source), duration_seconds
            )
            output_dir = Path(frame_root) / safe_video_key(video_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            paths = self._decode_opencv(
                capture,
                native_fps,
                total_native_frames,
                plan,
                output_dir,
                cv2,
            )
        finally:
            capture.release()

        expected = sum(clip.sampled_frame_count for clip in plan.clips)
        decoded = sum(len(items) for items in paths)
        incomplete_ratio = (expected - decoded) / expected
        decoder = "opencv"
        threshold = self.ffmpeg_fallback_min_incomplete_ratio
        if threshold is not None and incomplete_ratio >= threshold:
            fallback = self._decode_ffmpeg(source, plan, output_dir)
            fallback_decoded = sum(len(items) for items in fallback)
            if fallback_decoded > decoded:
                paths = fallback
                decoded = fallback_decoded
                incomplete_ratio = (expected - decoded) / expected
                decoder = "ffmpeg"

        clips = []
        for clip, frame_paths in zip(plan.clips, paths, strict=True):
            clips.append(
                replace(
                    clip,
                    frame_paths=frame_paths,
                    metadata={
                        **clip.metadata,
                        "decoder": decoder,
                        "expected_frames": clip.sampled_frame_count,
                        "decoded_frames": len(frame_paths),
                        "native_fps": native_fps,
                    },
                )
            )
        return replace(
            plan,
            clips=clips,
            metadata={
                **plan.metadata,
                "planning_source": "probed_video",
                "native_fps": native_fps,
                "total_native_frames": total_native_frames,
                "annotation_duration_seconds": annotation_duration_seconds,
                "duration_delta_seconds": (
                    None
                    if annotation_duration_seconds is None
                    else duration_seconds - annotation_duration_seconds
                ),
                "decoder": decoder,
                "decoded_sampled_frames": decoded,
                "incomplete_ratio": incomplete_ratio,
            },
        )

    def _decode_opencv(
        self,
        capture,
        native_fps: float,
        total_native_frames: int,
        plan: VgentVideoPlan,
        output_dir: Path,
        cv2,
    ) -> list[list[str]]:
        targets: dict[int, list[tuple[int, int]]] = {}
        for clip in plan.clips:
            for local_index, sample_index in enumerate(
                range(clip.sample_start_index, clip.sample_end_index)
            ):
                timestamp = (sample_index + 0.5) / plan.target_fps
                native_index = min(
                    total_native_frames - 1,
                    int(timestamp * native_fps),
                )
                targets.setdefault(native_index, []).append(
                    (clip.clip_index, local_index)
                )

        paths_by_slot: list[dict[int, str]] = [{} for _ in plan.clips]
        final_target = max(targets, default=-1)
        frame_number = 0
        while frame_number <= final_target:
            ok, frame = capture.read()
            if not ok:
                break
            for clip_index, local_index in targets.get(frame_number, []):
                destination = self._frame_path(output_dir, clip_index, local_index)
                if destination.is_file() or cv2.imwrite(
                    str(destination),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.opencv_jpeg_quality],
                ):
                    paths_by_slot[clip_index][local_index] = str(destination)
            frame_number += 1
        return [[slots[index] for index in sorted(slots)] for slots in paths_by_slot]

    def _decode_ffmpeg(
        self, source: Path, plan: VgentVideoPlan, output_dir: Path
    ) -> list[list[str]]:
        with TemporaryDirectory(
            prefix=f".{output_dir.name}.ffmpeg-", dir=output_dir.parent
        ) as temporary_name:
            temporary = Path(temporary_name)
            try:
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-i",
                        str(source),
                        "-an",
                        "-vf",
                        f"fps={plan.target_fps}",
                        "-q:v",
                        str(self.ffmpeg_jpeg_quality),
                        str(temporary / "%08d.jpg"),
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                return [[] for _ in plan.clips]
            expected = sum(clip.sampled_frame_count for clip in plan.clips)
            frame_files = sorted(temporary.glob("*.jpg"))[:expected]
            if result.returncode != 0 and not frame_files:
                return [[] for _ in plan.clips]
            paths: list[list[str]] = [[] for _ in plan.clips]
            for sample_index, frame in enumerate(frame_files):
                clip_index, local_index = divmod(
                    sample_index, self.config.frames_per_clip
                )
                if clip_index >= len(paths):
                    break
                destination = self._frame_path(output_dir, clip_index, local_index)
                shutil.copy2(frame, destination)
                paths[clip_index].append(str(destination))
            return paths

    @staticmethod
    def _frame_path(output_dir: Path, clip_index: int, frame_index: int) -> Path:
        clip_dir = output_dir / f"clip_{clip_index:05d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        return clip_dir / f"frame_{frame_index:03d}.jpg"


def safe_video_key(video_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", video_id).strip("._") or "video"
    digest = hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[:80]}_{digest}"


def video_manifest_path(root: str | Path, video_id: str) -> Path:
    return Path(root) / f"{safe_video_key(video_id)}.json"


def save_video_plan(plan: VgentVideoPlan, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_video_plan(path: str | Path) -> VgentVideoPlan:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return VgentVideoPlan.from_dict(payload)


def video_plan_cache_complete(plan: VgentVideoPlan) -> bool:
    return all(
        len(clip.frame_paths) == clip.sampled_frame_count
        and all(Path(path).is_file() for path in clip.frame_paths)
        for clip in plan.clips
    )
