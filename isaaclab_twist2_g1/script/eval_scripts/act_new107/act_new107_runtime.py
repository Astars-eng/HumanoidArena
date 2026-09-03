"""Optional EMA smoothing for ACT raw107 direct joint targets."""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np


EXPECTED_PREDICTED_CHUNK_STEPS = 25
EXPECTED_BODY_TARGET_DIM = 29


def ema_smooth_target(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    """Apply ``alpha * current + (1-alpha) * previous`` to one 29-D target."""

    previous_vector = np.asarray(previous, dtype=np.float32).reshape(-1)
    current_vector = np.asarray(current, dtype=np.float32).reshape(-1)
    if previous_vector.shape != (EXPECTED_BODY_TARGET_DIM,):
        raise ValueError(f"previous target must have shape (29,), got {previous_vector.shape}")
    if current_vector.shape != (EXPECTED_BODY_TARGET_DIM,):
        raise ValueError(f"current target must have shape (29,), got {current_vector.shape}")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value <= 1.0:
        raise ValueError(f"smooth alpha must be in (0, 1], got {alpha!r}")
    if not np.isfinite(previous_vector).all() or not np.isfinite(current_vector).all():
        raise ValueError("EMA targets contain NaN/Inf")
    return (alpha_value * current_vector + (1.0 - alpha_value) * previous_vector).astype(np.float32)


class ActNew107DirectRawSmoother:
    def __init__(self, action_provider: Any, *, smooth_alpha: float) -> None:
        alpha_value = float(smooth_alpha)
        if not np.isfinite(alpha_value) or not 0.0 < alpha_value <= 1.0:
            raise ValueError(f"smooth_alpha must be in (0, 1], got {smooth_alpha!r}")
        action_format = str(getattr(action_provider, "_vla_action_format", "")).strip().lower()
        if action_format != "raw107":
            raise RuntimeError(f"ACT raw107 smoother requires action_format=raw107, got {action_format!r}")
        body_source = str(getattr(action_provider, "_raw107_body_source", "")).strip().lower()
        if body_source != "direct_raw":
            raise RuntimeError("ACT raw107 smoothing requires SONIC_RAW107_BODY_SOURCE=direct_raw")

        self.action_provider = action_provider
        self.smooth_alpha = alpha_value
        self.previous_smoothed_target: np.ndarray | None = None
        self._original_on_env_reset = action_provider.on_env_reset
        self._original_run_raw107 = action_provider._run_gear_sonic_raw107_from_vla
        self._install_provider_hooks()

    def _install_provider_hooks(self) -> None:
        smoother = self

        def smoothed_on_env_reset(provider_self):
            result = smoother._original_on_env_reset()
            smoother.previous_smoothed_target = None
            return result

        def smoothed_run_raw107(provider_self) -> np.ndarray:
            target = smoother._original_run_raw107()
            return smoother.smooth_body_target(target)

        self.action_provider.on_env_reset = MethodType(smoothed_on_env_reset, self.action_provider)
        self.action_provider._run_gear_sonic_raw107_from_vla = MethodType(
            smoothed_run_raw107,
            self.action_provider,
        )
        setattr(self.action_provider, "_act_new107_direct_raw_smoother", self)

    def _current_body_joint_position(self) -> np.ndarray:
        robot = self.action_provider.env.scene["robot"].data
        current = robot.joint_pos[0, self.action_provider._sonic_idx]
        if hasattr(current, "detach"):
            current = current.detach()
        if hasattr(current, "cpu"):
            current = current.cpu()
        if hasattr(current, "numpy"):
            current = current.numpy()
        vector = np.asarray(current, dtype=np.float32).reshape(-1)
        if vector.shape != (EXPECTED_BODY_TARGET_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"current SONIC joint position must be finite (29,), got {vector.shape}")
        return vector.copy()

    def smooth_body_target(self, target: np.ndarray) -> np.ndarray:
        current_target = np.asarray(target, dtype=np.float32).reshape(-1)
        previous = self.previous_smoothed_target
        if previous is None:
            previous = self._current_body_joint_position()
        smoothed = ema_smooth_target(previous, current_target, self.smooth_alpha)
        self.previous_smoothed_target = smoothed.copy()
        self.action_provider._latest_decoder_target = smoothed.copy()
        return smoothed


def configure_act_new107_smoothing(
    action_provider: Any,
    *,
    smooth_alpha: float,
) -> ActNew107DirectRawSmoother:
    smoother = ActNew107DirectRawSmoother(action_provider, smooth_alpha=smooth_alpha)
    print(
        "[act_new107] direct_raw EMA enabled "
        f"alpha={smoother.smooth_alpha:.3f} formula=alpha*current+(1-alpha)*previous"
    )
    return smoother
