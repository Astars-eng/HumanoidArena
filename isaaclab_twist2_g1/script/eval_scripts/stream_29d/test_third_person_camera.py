from __future__ import annotations

import math
from pathlib import Path

import pytest

from third_person_camera import compute_follow_camera_pose, world_camera_video_dir


def test_world_camera_video_dir_preserves_result_bucket() -> None:
    assert world_camera_video_dir(Path("/tmp/run/videos/success")) == Path(
        "/tmp/run/videos/world_camera/success"
    )
    assert world_camera_video_dir(Path("/tmp/run/videos/failure")) == Path(
        "/tmp/run/videos/world_camera/failure"
    )


def test_follow_camera_tracks_robot_translation_and_keeps_full_body_target() -> None:
    eye, target = compute_follow_camera_pose(
        (-3.7, -2.2, 0.8),
        (1.0, 0.0, 0.0, 0.0),
        distance=4.0,
        camera_height=2.2,
        target_height=0.9,
        lateral_offset=1.25,
    )

    assert eye == pytest.approx((-7.7, -0.95, 3.0))
    assert target == pytest.approx((-3.7, -2.2, 1.7))


def test_follow_camera_rotates_with_robot_heading() -> None:
    half_yaw = math.pi / 4.0
    eye, target = compute_follow_camera_pose(
        (10.0, 5.0, 0.5),
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        distance=4.0,
        camera_height=2.2,
        target_height=0.9,
        lateral_offset=1.25,
    )

    assert eye == pytest.approx((8.75, 1.0, 2.7))
    assert target == pytest.approx((10.0, 5.0, 1.4))


@pytest.mark.parametrize(
    ("distance", "camera_height", "target_height"),
    [(0.0, 2.2, 0.9), (-1.0, 2.2, 0.9), (4.0, 0.9, 0.9)],
)
def test_follow_camera_rejects_invalid_framing(
    distance: float, camera_height: float, target_height: float
) -> None:
    with pytest.raises(ValueError):
        compute_follow_camera_pose(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            distance=distance,
            camera_height=camera_height,
            target_height=target_height,
            lateral_offset=0.0,
        )
