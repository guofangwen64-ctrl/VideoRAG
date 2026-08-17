"""On-demand dense frame extraction for the QA reader stage."""
from __future__ import annotations

import subprocess
from pathlib import Path

from medhorizon_videorag.core.schemas import Chunk


class FineFrameExtractor:
    """Extract and cache uniformly spaced evidence frames for retrieved chunks.

    These frames are deliberately separate from index-time frames: dense
    sampling is paid only for the few chunks selected for one QA example.
    """

    def __init__(self, frame_root: str | Path, frames_per_chunk: int = 16) -> None:
        if frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be positive")
        self.frame_root = Path(frame_root)
        self.frames_per_chunk = frames_per_chunk

    def extract(self, chunk: Chunk) -> list[str]:
        return self._extract(chunk.video_id, chunk.video_path, chunk.start_seconds, chunk.end_seconds, chunk.id)

    def extract_window(self, video_id: str, video_path: str, start_seconds: float, end_seconds: float) -> list[str]:
        """Extract frames across an entire user-specified temporal range."""
        cache_key = f"temporal_{start_seconds:010.3f}_{end_seconds:010.3f}".replace(".", "_")
        return self._extract(video_id, video_path, start_seconds, end_seconds, cache_key)

    def _extract(
        self, video_id: str, video_path: str, start_seconds: float, end_seconds: float, cache_key: str,
    ) -> list[str]:
        if end_seconds <= start_seconds:
            raise ValueError("Reader evidence window must have positive duration")
        output_dir = self.frame_root / video_id / cache_key
        existing = sorted(output_dir.glob("*.jpg"))
        if len(existing) >= self.frames_per_chunk:
            return [str(path) for path in existing[:self.frames_per_chunk]]

        output_dir.mkdir(parents=True, exist_ok=True)
        for path in existing:
            path.unlink()
        duration = end_seconds - start_seconds
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-ss", str(start_seconds),
            "-i", video_path, "-t", str(duration), "-an",
            "-vf", f"fps={self.frames_per_chunk / duration}", "-frames:v", str(self.frames_per_chunk),
            "-q:v", "2", "-y", str(output_dir / "%02d.jpg"),
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
        except FileNotFoundError as error:
            raise RuntimeError("FFmpeg is required for Reader frame extraction") from error
        frames = sorted(output_dir.glob("*.jpg"))
        if result.returncode != 0 and not frames:
            message = result.stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"Could not extract Reader frames for {cache_key}: {message}")
        return [str(path) for path in frames]
