"""Task-agnostic interface adapter for SONIC raw-action LeRobot policies.

This module deliberately contains only robot/policy interface conversion.  It
does not inspect task state, object poses, rewards, or language instructions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SONIC_RAW_POLICY_STATE_DIM = 64
SONIC_RAW_POLICY_ACTION_DIM = 107
SONIC_RAW_BODY_ACTION_DIM = 29
SONIC_RAW_ENCODER_TOKEN_DIM = 64
SONIC_RAW_HAND_ACTION_DIM = 7

SONIC_RAW_STATE_KEYS = (
    "observation.joint_pos",
    "observation.joint_vel",
    "observation.ang_vel_b",
    "observation.gravity",
)


def _finite_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return vector


@dataclass(frozen=True)
class SonicRawPolicyAction:
    """Named views of the checkpoint's concatenated 107-D output."""

    body_raw: np.ndarray
    encoder_token: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray


def split_sonic_raw_policy_action(action: np.ndarray) -> SonicRawPolicyAction:
    """[interface conversion] Split 107-D policy output without modification."""

    vector = _finite_vector(action, SONIC_RAW_POLICY_ACTION_DIM, "SONIC raw policy action")
    body_end = SONIC_RAW_BODY_ACTION_DIM
    token_end = body_end + SONIC_RAW_ENCODER_TOKEN_DIM
    left_end = token_end + SONIC_RAW_HAND_ACTION_DIM
    return SonicRawPolicyAction(
        body_raw=vector[:body_end].copy(),
        encoder_token=vector[body_end:token_end].copy(),
        left_hand=vector[token_end:left_end].copy(),
        right_hand=vector[left_end:].copy(),
    )


def sonic_raw_body_to_joint_targets(
    body_raw: np.ndarray,
    *,
    action_scale: np.ndarray,
    default_joint_pos: np.ndarray,
) -> np.ndarray:
    """[interface conversion] Apply the native SONIC decoder post-processing.

    The source dataset stores ``decoder_raw_action``.  SONIC deploy converts it
    to an absolute radian target with exactly ``raw * scale + default`` and does
    not clip the raw policy output in this layer.
    """

    raw = _finite_vector(body_raw, SONIC_RAW_BODY_ACTION_DIM, "SONIC raw body action")
    scale = _finite_vector(action_scale, SONIC_RAW_BODY_ACTION_DIM, "SONIC action scale")
    default = _finite_vector(default_joint_pos, SONIC_RAW_BODY_ACTION_DIM, "SONIC default joint pose")
    target = raw * scale + default
    if not np.isfinite(target).all():
        raise ValueError("converted SONIC joint target contains NaN or Inf")
    return target.astype(np.float32, copy=False)


def build_sonic_raw_policy_observation(
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    ang_vel_b: np.ndarray,
    gravity: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """[interface conversion] Build the checkpoint's 64-D proprioception.

    Joint arrays must be supplied in the same SONIC IsaacLab order recorded by
    ``SonicActionProvider``.  Values and units are passed through unchanged:
    joint position in rad, joint velocity/angular velocity in rad/s, and unit
    projected gravity in the robot base frame.
    """

    components = {
        "observation.joint_pos": _finite_vector(joint_pos, 29, "joint_pos").copy(),
        "observation.joint_vel": _finite_vector(joint_vel, 29, "joint_vel").copy(),
        "observation.ang_vel_b": _finite_vector(ang_vel_b, 3, "ang_vel_b").copy(),
        "observation.gravity": _finite_vector(gravity, 3, "gravity").copy(),
    }
    state = np.concatenate([components[key] for key in SONIC_RAW_STATE_KEYS]).astype(
        np.float32, copy=False
    )
    if state.shape != (SONIC_RAW_POLICY_STATE_DIM,):
        raise RuntimeError(f"unexpected SONIC raw policy state shape: {state.shape}")
    return state, components
