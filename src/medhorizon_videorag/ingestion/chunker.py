from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Iterator

from medhorizon_videorag.core.schemas import Chunk


@dataclass
class VideoChunker:
    """Create fixed temporal windows and sample their frames in one decode pass."""

    duration_seconds: float = 30.0
    stride_seconds: float = 30.0
    frames_per_chunk: int = 8

    def chunk(
        self, video_id: str, video_path: str, frame_root: str | Path,
        ffmpeg_fallback_min_incomplete_ratio: float | None = None,
    ) -> list[Chunk]:
        """Decode a video once, sampling frames without random seeks.

        A chunk is emitted even if a damaged stream prevents some frames from
        being decoded. Its ``frame_paths`` then exposes the missing evidence to
        downstream validation rather than silently inventing a replacement.
        """
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Install video dependencies: pip install -e '.[video]'") from error

        source = Path(video_path)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Cannot open video: {source}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError(f"Video has no decodable frames: {source}")
            total_seconds = total_frames / fps
            windows = list(self.time_windows(total_seconds))
            output_dir = Path(frame_root) / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            sampled_paths = self._decode_sampled_frames(capture, fps, windows, output_dir)
        finally:
            capture.release()

        incomplete_ratio = sum(len(paths) < max(1, self.frames_per_chunk) for paths in sampled_paths) / len(windows)
        decoder = "opencv"
        if ffmpeg_fallback_min_incomplete_ratio is not None and incomplete_ratio >= ffmpeg_fallback_min_incomplete_ratio:
            fallback_paths = self._ffmpeg_sample(source, windows, output_dir)
            fallback_ratio = sum(len(paths) < max(1, self.frames_per_chunk) for paths in fallback_paths) / len(windows)
            if fallback_ratio < incomplete_ratio:
                sampled_paths, incomplete_ratio, decoder = fallback_paths, fallback_ratio, "ffmpeg"

        return [Chunk(
            id=f"{video_id}_{index:05d}", video_id=video_id, video_path=str(source),
            start_seconds=round(start, 3), end_seconds=round(end, 3),
            frame_paths=sampled_paths[index],
            metadata={
                "expected_frames": max(1, self.frames_per_chunk), "decoded_frames": len(sampled_paths[index]),
                "decoder": decoder,
            },
        ) for index, (start, end) in enumerate(windows)]

    def time_windows(self, total_seconds: float) -> Iterator[tuple[float, float]]:
        """Yield fixed windows; retain a final shorter tail window."""
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero")
        if self.stride_seconds <= 0:
            raise ValueError("stride_seconds must be greater than zero")
        start = 0.0
        while start < total_seconds:
            yield start, min(start + self.duration_seconds, total_seconds)
            start += self.stride_seconds

    def _decode_sampled_frames(
        self, capture, fps: float, windows: list[tuple[float, float]], output_dir: Path
    ) -> list[list[str]]:
        """Read forward once and write frames closest to uniformly spaced targets."""
        import cv2

        target_map: dict[int, tuple[int, int]] = {}
        sample_count = max(1, self.frames_per_chunk)
        for chunk_index, (start, end) in enumerate(windows):
            for sample_index in range(sample_count):
                timestamp = start + (end - start) * (sample_index + 0.5) / sample_count
                target_map[int(round(timestamp * fps))] = (chunk_index, sample_index)

        paths: list[list[str]] = [[] for _ in windows]
        final_target = max(target_map, default=-1)
        frame_number = 0
        while frame_number <= final_target:
            ok, frame = capture.read()
            if not ok:
                # Broken/truncated streams may end before their advertised frame count.
                break
            target = target_map.get(frame_number)
            if target:
                chunk_index, sample_index = target
                destination = output_dir / f"{chunk_index:05d}_{sample_index:02d}.jpg"
                if cv2.imwrite(str(destination), frame):
                    paths[chunk_index].append(str(destination))
            frame_number += 1
        return paths

    def _ffmpeg_sample(
        self, source: Path, windows: list[tuple[float, float]], output_dir: Path
    ) -> list[list[str]]:
        """Fallback for streams OpenCV cannot fully decode.

        FFmpeg decodes the complete stream once at a fixed sampling rate. It is
        intentionally used only for severely incomplete OpenCV results.
        """
        sample_count = max(1, self.frames_per_chunk)
        temporary = output_dir.parent / f".{output_dir.name}.ffmpeg-tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-an",
                    "-vf", f"fps={sample_count / self.duration_seconds}", "-q:v", "2",
                    str(temporary / "%08d.jpg"),
                ],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            frame_files = sorted(temporary.glob("*.jpg"))
            if result.returncode != 0 and not frame_files:
                return [[] for _ in windows]
            paths: list[list[str]] = [[] for _ in windows]
            for index, frame in enumerate(frame_files):
                chunk_index, sample_index = divmod(index, sample_count)
                if chunk_index >= len(windows):
                    break
                destination = output_dir / f"{chunk_index:05d}_{sample_index:02d}.jpg"
                shutil.move(str(frame), destination)
                paths[chunk_index].append(str(destination))
            return paths
        except FileNotFoundError:
            return [[] for _ in windows]
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
