from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from medhorizon_videorag.core.schemas import Chunk


@dataclass
class VideoChunker:
    duration_seconds: float = 16.0
    stride_seconds: float = 8.0
    frames_per_chunk: int = 8

    def chunk(self, video_id: str, video_path: str) -> list[Chunk]:
        """Decode a video into temporal chunks and uniformly sampled frames."""
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Install video dependencies: pip install -e '.[video]'") from error

        source = Path(video_path)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / fps
        capture.release()

        chunks_dir = source.parent / ".frames" / video_id
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[Chunk] = []
        start = 0.0
        index = 0
        while start < total_seconds:
            end = min(start + self.duration_seconds, total_seconds)
            frame_paths = list(self._sample_frames(source, fps, start, end, index, chunks_dir))
            chunks.append(Chunk(
                id=f"{video_id}_{index:05d}", video_id=video_id, video_path=str(source),
                start_seconds=round(start, 3), end_seconds=round(end, 3), frame_paths=frame_paths,
            ))
            index += 1
            start += self.stride_seconds
        return chunks

    def _sample_frames(self, source: Path, fps: float, start: float, end: float, index: int, output_dir: Path) -> Iterator[str]:
        import cv2

        capture = cv2.VideoCapture(str(source))
        sample_count = max(1, self.frames_per_chunk)
        for frame_index in range(sample_count):
            timestamp = start + (end - start) * (frame_index + 0.5) / sample_count
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
            ok, frame = capture.read()
            if ok:
                target = output_dir / f"{index:05d}_{frame_index:02d}.jpg"
                cv2.imwrite(str(target), frame)
                yield str(target)
        capture.release()
