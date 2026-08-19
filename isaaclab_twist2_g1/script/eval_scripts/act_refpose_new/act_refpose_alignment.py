"""ACT alignment for the raw-proprioception RefPose v3.1 checkpoints.

The checkpoint contract is intentionally kept local to this evaluator:

* input state: joint_pos(29), joint_vel(29), ang_vel_b(3), gravity(3)
* observation history: [t-9, ..., t] at the 50 Hz control rate
* output action: 40-D RefPose v3.1
* prediction: 25-step ACT chunk, with a configurable prefix executed before
  replanning

The shared SONIC provider has multiple 64-D state protocols.  Patching its
HTTP client's ``infer_chunk`` method is not sufficient because the provider
may select a different request path based on environment variables.  This
module therefore patches the provider instance's chunk-fetch boundary and
always constructs the checkpoint's raw proprioception explicitly.
"""

from __future__ import annotations

import base64
import time
from collections import deque
from types import MethodType
from typing import Any

import numpy as np

STATE_COMPONENT_SPECS = (
    ("observation.joint_pos", 29),
    ("observation.joint_vel", 29),
    ("observation.ang_vel_b", 3),
    ("observation.gravity", 3),
)
EXPECTED_STATE_DIM = sum(width for _, width in STATE_COMPONENT_SPECS)
EXPECTED_HISTORY_STEPS = 10
EXPECTED_ACTION_DIM = 40
EXPECTED_PREDICTED_CHUNK_STEPS = 25


class ActRefposeAlignment:
    def __init__(
        self,
        action_provider: Any,
        *,
        history_steps: int,
        execute_steps: int,
        record_full_chunks: bool,
    ) -> None:
        if history_steps != EXPECTED_HISTORY_STEPS:
            raise ValueError(
                f"This checkpoint requires history_steps={EXPECTED_HISTORY_STEPS}, got {history_steps}"
            )
        if execute_steps <= 0:
            raise ValueError(f"execute_steps must be positive, got {execute_steps}")
        if execute_steps > EXPECTED_PREDICTED_CHUNK_STEPS:
            raise ValueError(
                f"execute_steps must not exceed the checkpoint chunk size "
                f"{EXPECTED_PREDICTED_CHUNK_STEPS}, got {execute_steps}"
            )

        client = getattr(action_provider, "_lerobot_http_client", None)
        if client is None:
            raise RuntimeError(
                "ACT RefPose alignment requires the remote LeRobot HTTP client"
            )
        action_format = (
            str(getattr(action_provider, "_vla_action_format", "")).strip().lower()
        )
        if action_format != "semantic_v3":
            raise RuntimeError(
                "ACT RefPose alignment requires SONIC_VLA_ACTION_FORMAT=semantic_v3 "
                f"for the checkpoint's 40-D output, got {action_format!r}"
            )

        self.action_provider = action_provider
        self.client = client
        self.history_steps = int(history_steps)
        self.execute_steps = int(execute_steps)
        self.record_full_chunks = bool(record_full_chunks)
        self.state_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.previous_executed_last_action: np.ndarray | None = None
        self.infer_index = 0

        self._original_get_action = action_provider.get_action
        self._original_on_env_reset = action_provider.on_env_reset
        self._original_fetch_action_chunk = action_provider._fetch_lerobot_action_chunk
        self._install_provider_hooks()

    def _install_provider_hooks(self) -> None:
        alignment = self

        def aligned_get_action(provider_self, env):
            # Capture before inference/action application.  This matches the
            # recorder's robot_*_before_decimation observations.
            alignment.capture_current_state()
            return alignment._original_get_action(env)

        def aligned_on_env_reset(provider_self):
            result = alignment._original_on_env_reset()
            alignment.reset_local_history()
            return result

        def aligned_fetch_action_chunk(provider_self) -> np.ndarray:
            return alignment.fetch_action_chunk()

        self.action_provider.get_action = MethodType(
            aligned_get_action, self.action_provider
        )
        self.action_provider.on_env_reset = MethodType(
            aligned_on_env_reset, self.action_provider
        )
        self.action_provider._fetch_lerobot_action_chunk = MethodType(
            aligned_fetch_action_chunk,
            self.action_provider,
        )
        self.action_provider._act_refpose_alignment = self

    def reset_local_history(self) -> None:
        self.state_history.clear()
        self.previous_executed_last_action = None
        self.infer_index = 0

    @staticmethod
    def _validate_raw_state(
        state: np.ndarray,
        components: dict[str, np.ndarray],
    ) -> np.ndarray:
        flat_state = np.asarray(state, dtype=np.float32).reshape(-1)
        if flat_state.shape != (EXPECTED_STATE_DIM,):
            raise ValueError(
                f"ACT RefPose raw state must have shape {(EXPECTED_STATE_DIM,)}, got {flat_state.shape}"
            )
        if not np.isfinite(flat_state).all():
            raise ValueError("ACT RefPose raw state contains NaN/Inf")

        ordered_components = []
        for key, width in STATE_COMPONENT_SPECS:
            if key not in components:
                raise KeyError(f"ACT RefPose raw observation is missing {key!r}")
            value = np.asarray(components[key], dtype=np.float32).reshape(-1)
            if value.shape != (width,):
                raise ValueError(f"{key} must have shape {(width,)}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains NaN/Inf")
            ordered_components.append(value)

        reconstructed = np.concatenate(ordered_components).astype(
            np.float32, copy=False
        )
        if not np.array_equal(flat_state, reconstructed):
            raise ValueError(
                "ACT RefPose flat state does not match the checkpoint component order "
                "joint_pos+joint_vel+ang_vel_b+gravity"
            )
        return flat_state

    def _read_current_raw_state(self) -> np.ndarray:
        state, components = self.action_provider._build_lerobot_raw107_observation()
        return self._validate_raw_state(state, components)

    def capture_current_state(self) -> None:
        self.state_history.append(self._read_current_raw_state().copy())

    def _stack_history(self, fallback_state: np.ndarray) -> np.ndarray:
        fallback = np.asarray(fallback_state, dtype=np.float32).reshape(-1)
        if fallback.shape != (EXPECTED_STATE_DIM,):
            raise ValueError(f"Unexpected fallback state shape: {fallback.shape}")
        states = list(self.state_history)
        if not states:
            states = [fallback.copy()]
        if any(state.shape != (EXPECTED_STATE_DIM,) for state in states):
            raise ValueError(
                "ACT RefPose state history contains inconsistent dimensions"
            )
        if len(states) < self.history_steps:
            states = [
                states[0].copy() for _ in range(self.history_steps - len(states))
            ] + states
        history = np.stack(states[-self.history_steps :], axis=0).astype(
            np.float32, copy=False
        )
        expected_shape = (self.history_steps, EXPECTED_STATE_DIM)
        if history.shape != expected_shape:
            raise ValueError(
                f"Expected ACT RefPose history shape {expected_shape}, got {history.shape}"
            )
        return history

    @staticmethod
    def _decode_action_chunk(response: dict[str, Any]) -> np.ndarray:
        action_chunk = response.get("action_chunk")
        if action_chunk is None:
            raise RuntimeError("ACT RefPose server response has no action_chunk")
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        expected_shape = (EXPECTED_PREDICTED_CHUNK_STEPS, EXPECTED_ACTION_DIM)
        if chunk.shape != expected_shape:
            raise RuntimeError(
                f"Expected ACT action chunk shape {expected_shape}, got {chunk.shape}"
            )
        if not np.isfinite(chunk).all():
            raise RuntimeError("ACT RefPose action chunk contains NaN/Inf")
        return chunk

    def fetch_action_chunk(self) -> np.ndarray:
        fallback_state = self._read_current_raw_state()
        if not self.state_history:
            self.state_history.append(fallback_state.copy())
        history = self._stack_history(fallback_state)
        rgb = np.asarray(self.action_provider._get_front_camera_rgb_for_vla())
        robot_type = str(
            getattr(self.action_provider, "_lerobot_robot_type", "")
        ).strip()
        task = getattr(self.action_provider, "task_name", None)
        task_name = None if task is None else str(task).strip()

        payload: dict[str, Any] = {
            "observation": {
                "images": {
                    "front": {
                        "shape": list(rgb.shape),
                        "dtype": str(rgb.dtype),
                        "data_b64": base64.b64encode(rgb.tobytes()).decode("ascii"),
                    }
                },
                # Oldest to newest: config obs_delta_sequence=[9,8,...,0].
                "state": history.tolist(),
            },
            "robot_type": robot_type,
            "return_chunk": True,
        }
        if task_name:
            payload["task"] = task_name

        response = self.client._post_json("/infer", payload)
        full_chunk = self._decode_action_chunk(response)
        executed_chunk = full_chunk[: self.execute_steps].copy()
        boundary_l2 = None
        if self.previous_executed_last_action is not None:
            boundary_l2 = float(
                np.linalg.norm(full_chunk[0] - self.previous_executed_last_action)
            )
        self.previous_executed_last_action = executed_chunk[-1].copy()

        trace: dict[str, Any] = {
            "event": "act_refpose_new_infer_chunk",
            "timestamp": time.time(),
            "step_idx": int(self.infer_index),
            "robot_type": robot_type,
            "task": task_name,
            "input_contract": "joint_pos29+joint_vel29+ang_vel_b3+gravity3",
            "history_order": "t-9_to_t"
            if self.history_steps == 10
            else "oldest_to_current",
            "front_rgb_shape": list(rgb.shape),
            "observation_state_shape": list(history.shape),
            "observation_state_first": history[0].tolist(),
            "observation_state_last": history[-1].tolist(),
            "predicted_chunk_size": int(full_chunk.shape[0]),
            "executed_chunk_size": int(executed_chunk.shape[0]),
            "first_action": full_chunk[0].tolist(),
            "executed_last_action": executed_chunk[-1].tolist(),
            "predicted_last_action": full_chunk[-1].tolist(),
            "boundary_l2": boundary_l2,
            "root_xy_norm_mean": float(
                np.linalg.norm(full_chunk[:, 0:2], axis=1).mean()
            ),
            "action_abs_mean": float(np.abs(full_chunk).mean()),
        }
        if self.record_full_chunks:
            trace["action_chunk"] = full_chunk.tolist()
        self.client._append_trace(trace)
        self.client._trace_step_idx += 1
        self.infer_index += 1
        return executed_chunk


def configure_act_refpose_alignment(
    action_provider: Any,
    *,
    history_steps: int,
    execute_steps: int,
    record_full_chunks: bool,
) -> ActRefposeAlignment:
    alignment = ActRefposeAlignment(
        action_provider,
        history_steps=history_steps,
        execute_steps=execute_steps,
        record_full_chunks=record_full_chunks,
    )
    print(
        "[act_refpose_new] checkpoint alignment enabled "
        f"input=raw64 history={alignment.history_steps}x@control-rate "
        f"output={EXPECTED_PREDICTED_CHUNK_STEPS}x{EXPECTED_ACTION_DIM} "
        f"execute_prefix={alignment.execute_steps} "
        f"record_full_chunks={int(alignment.record_full_chunks)}"
    )
    return alignment
