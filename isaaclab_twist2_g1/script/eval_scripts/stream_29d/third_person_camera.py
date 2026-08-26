"""Task-independent geometry for the stream_29d third-person camera."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path


def world_camera_video_dir(front_video_dir: Path) -> Path:
    """Return the sibling view directory while preserving the outcome bucket."""
    front_video_dir = Path(front_video_dir)
    return front_video_dir.parent / "world_camera" / front_video_dir.name


def compute_follow_camera_pose(
    root_position: Sequence[float],
    root_quaternion_wxyz: Sequence[float],
    *,
    distance: float,
    camera_height: float,
    target_height: float,
    lateral_offset: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return eye/target points for a rear three-quarter robot-follow view."""
    if len(root_position) != 3 or len(root_quaternion_wxyz) != 4:
        raise ValueError("root pose must contain position[3] and quaternion_wxyz[4]")
    if distance <= 0.0 or camera_height <= target_height:
        raise ValueError("camera requires distance > 0 and camera_height > target_height")

    root_x, root_y, root_z = (float(value) for value in root_position)
    quat_w, quat_x, quat_y, quat_z = (float(value) for value in root_quaternion_wxyz)
    values = (
        root_x,
        root_y,
        root_z,
        quat_w,
        quat_x,
        quat_y,
        quat_z,
        distance,
        camera_height,
        target_height,
        lateral_offset,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera pose inputs must be finite")

    forward_x = 1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z)
    forward_y = 2.0 * (quat_x * quat_y + quat_w * quat_z)
    forward_norm = math.hypot(forward_x, forward_y)
    if forward_norm < 1.0e-6:
        forward_x, forward_y = 1.0, 0.0
    else:
        forward_x /= forward_norm
        forward_y /= forward_norm

    right_x, right_y = -forward_y, forward_x
    eye = (
        root_x - distance * forward_x + lateral_offset * right_x,
        root_y - distance * forward_y + lateral_offset * right_y,
        root_z + camera_height,
    )
    target = (root_x, root_y, root_z + target_height)
    return eye, target
