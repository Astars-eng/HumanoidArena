from __future__ import annotations

from pathlib import Path

from third_person_camera import world_camera_video_dir


def test_world_camera_video_dir_preserves_result_bucket() -> None:
    assert world_camera_video_dir(Path("/tmp/run/videos/success")) == Path(
        "/tmp/run/videos/world_camera/success"
    )
    assert world_camera_video_dir(Path("/tmp/run/videos/failure")) == Path(
        "/tmp/run/videos/world_camera/failure"
    )
