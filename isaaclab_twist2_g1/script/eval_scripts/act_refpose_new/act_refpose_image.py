"""Image preprocessing contract shared by the RefPose evaluator and tests."""

from __future__ import annotations

import cv2
import numpy as np


def resize_rgb_to_policy_shape(
    image: np.ndarray,
    expected_chw: tuple[int, ...],
) -> np.ndarray:
    """Match the deterministic resize used by convert_raw_to_refpose.py."""
    if len(expected_chw) != 3 or expected_chw[0] != 3:
        raise ValueError(f"Expected a 3-channel CHW policy image shape, got {expected_chw}")
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HWC RGB/RGBA image, got {image.shape}")

    target_height, target_width = int(expected_chw[1]), int(expected_chw[2])
    rgb = np.ascontiguousarray(image[..., :3])
    if rgb.shape[:2] == (target_height, target_width):
        return rgb

    # The dataset converter changes 4:3 to 1:1 directly and uses INTER_AREA
    # whenever both source dimensions are at least as large as the target.
    interpolation = (
        cv2.INTER_AREA
        if rgb.shape[0] >= target_height and rgb.shape[1] >= target_width
        else cv2.INTER_LINEAR
    )
    return np.ascontiguousarray(
        cv2.resize(rgb, (target_width, target_height), interpolation=interpolation)
    )
