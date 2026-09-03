"""Task-independent paths for the stream_29d third-person camera."""

from __future__ import annotations

from pathlib import Path


def world_camera_video_dir(front_video_dir: Path) -> Path:
    """Return the sibling view directory while preserving the outcome bucket."""
    front_video_dir = Path(front_video_dir)
    return front_video_dir.parent / "world_camera" / front_video_dir.name
