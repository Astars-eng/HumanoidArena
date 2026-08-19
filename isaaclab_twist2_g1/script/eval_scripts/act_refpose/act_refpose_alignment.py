"""Isolated ACT ref-pose timing alignment for the DoubleDesk evaluator.

The shared SONIC action provider intentionally remains unchanged.  This module
patches one provider instance created by ``sim_eval_vla.py`` so the policy sees
the same 50 Hz state history used during training and so only a configurable
prefix of each predicted action chunk is executed before replanning.
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
EXPECTED_ACTION_DIM = 52
EXPECTED_SEMANTIC_BODY_DIM = 38
EXPECTED_SEMANTIC_ACTION_DIM = 40
EXPECTED_HAND_DIM = 7
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
                f"execute_steps must not exceed chunk size {EXPECTED_PREDICTED_CHUNK_STEPS}, "
                f"got {execute_steps}"
            )

        client = getattr(action_provider, "_lerobot_http_client", None)
        if client is None:
            raise RuntimeError("ACT ref-pose alignment requires the remote LeRobot HTTP client")
        action_format = str(getattr(action_provider, "_vla_action_format", "")).strip().lower()
        if action_format != "semantic_v3":
            raise RuntimeError(
                "ACT ref-pose alignment requires the semantic_v3 body runtime, "
                f"got {action_format!r}"
            )

        self.action_provider = action_provider
        self.client = client
        self.history_steps = int(history_steps)
        self.execute_steps = int(execute_steps)
        self.record_full_chunks = bool(record_full_chunks)
        self.state_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self.continuous_hand_queue: deque[np.ndarray] = deque()
        self.previous_executed_last_action: np.ndarray | None = None
        self.infer_index = 0

        self._original_get_action = action_provider.get_action
        self._original_on_env_reset = action_provider.on_env_reset
        self._original_apply_semantic_action = action_provider._apply_lerobot_semantic_action
        self._install_provider_hooks()

    def _install_provider_hooks(self) -> None:
        alignment = self

        def aligned_get_action(provider_self, env):
            alignment.capture_current_state()
            return alignment._original_get_action(env)

        def aligned_on_env_reset(provider_self):
            result = alignment._original_on_env_reset()
            alignment.reset_local_history()
            return result

        def aligned_fetch_lerobot_action_chunk(provider_self) -> np.ndarray:
            return alignment.fetch_lerobot_action_chunk()

        def aligned_apply_semantic_action(provider_self, action: np.ndarray) -> None:
            alignment.apply_semantic_action(action)

        self.action_provider.get_action = MethodType(aligned_get_action, self.action_provider)
        self.action_provider.on_env_reset = MethodType(aligned_on_env_reset, self.action_provider)
        self.action_provider._fetch_lerobot_action_chunk = MethodType(
            aligned_fetch_lerobot_action_chunk,
            self.action_provider,
        )
        self.action_provider._apply_lerobot_semantic_action = MethodType(
            aligned_apply_semantic_action,
            self.action_provider,
        )
        setattr(self.action_provider, "_act_refpose_alignment", self)

    def reset_local_history(self) -> None:
        self.state_history.clear()
        self.continuous_hand_queue.clear()
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
                f"ACT ref-pose raw state must have shape {(EXPECTED_STATE_DIM,)}, got {flat_state.shape}"
            )
        ordered_components = []
        for key, width in STATE_COMPONENT_SPECS:
            if key not in components:
                raise KeyError(f"ACT ref-pose raw observation is missing {key!r}")
            value = np.asarray(components[key], dtype=np.float32).reshape(-1)
            if value.shape != (width,):
                raise ValueError(f"{key} must have shape {(width,)}, got {value.shape}")
            ordered_components.append(value)
        reconstructed = np.concatenate(ordered_components).astype(np.float32, copy=False)
        if not np.isfinite(flat_state).all() or not np.isfinite(reconstructed).all():
            raise ValueError("ACT ref-pose raw state contains NaN/Inf")
        if not np.array_equal(flat_state, reconstructed):
            raise ValueError(
                "ACT ref-pose state order must be joint_pos+joint_vel+ang_vel_b+gravity"
            )
        return flat_state

    def _read_current_raw_state(self) -> np.ndarray:
        state, components = self.action_provider._build_lerobot_raw107_observation()
        return self._validate_raw_state(state, components)

    def capture_current_state(self) -> None:
        self.state_history.append(self._read_current_raw_state().copy())

    def _stack_history(self, fallback_state: np.ndarray) -> np.ndarray:
        fallback = np.asarray(fallback_state, dtype=np.float32).reshape(-1)
        states = list(self.state_history)
        if not states:
            states = [fallback.copy()]
        if any(state.shape != fallback.shape for state in states):
            raise ValueError("ACT ref-pose state history contains inconsistent dimensions")
        if len(states) < self.history_steps:
            states = [states[0].copy() for _ in range(self.history_steps - len(states))] + states
        history = np.stack(states[-self.history_steps :], axis=0).astype(np.float32, copy=False)
        if history.shape != (self.history_steps, fallback.size):
            raise ValueError(f"Unexpected aligned state history shape: {history.shape}")
        return history

    @staticmethod
    def _decode_action_chunk(response: dict[str, Any]) -> np.ndarray:
        action_chunk = response.get("action_chunk")
        if action_chunk is None and "action" in response:
            action_chunk = [response["action"]]
        if action_chunk is None:
            raise RuntimeError("ACT ref-pose server response has no action_chunk/action")
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        expected_shape = (EXPECTED_PREDICTED_CHUNK_STEPS, EXPECTED_ACTION_DIM)
        if chunk.shape != expected_shape:
            raise RuntimeError(f"Expected ACT ref-pose chunk shape {expected_shape}, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise RuntimeError("ACT ref-pose action chunk contains NaN/Inf")
        return chunk

    def apply_semantic_action(self, action: np.ndarray) -> None:
        if not self.continuous_hand_queue:
            raise RuntimeError("ACT ref-pose continuous hand queue is empty")
        hand_targets = self.continuous_hand_queue.popleft()
        self._original_apply_semantic_action(action)
        self.action_provider._left_hand_target[:] = hand_targets[:EXPECTED_HAND_DIM]
        self.action_provider._right_hand_target[:] = hand_targets[EXPECTED_HAND_DIM:]

    def fetch_lerobot_action_chunk(self) -> np.ndarray:
        """Fetch and adapt one RefPose chunk before shared shape validation runs."""
        rgb = self.action_provider._get_front_camera_rgb_for_vla()
        state, components = self.action_provider._build_lerobot_raw107_observation()
        state = self._validate_raw_state(state, components)
        return self.infer_chunk(
            front_rgb=rgb,
            observation_state=state,
            robot_type=self.action_provider._lerobot_robot_type,
            task=self.action_provider.task_name,
            observation_components=components,
        )

    def infer_chunk(
        self,
        *,
        front_rgb: np.ndarray,
        observation_state: np.ndarray,
        robot_type: str,
        task: str | None,
        observation_components: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        rgb = np.asarray(front_rgb)
        if observation_components is None:
            raise ValueError("ACT ref-pose inference requires raw64 observation components")
        fallback_state = self._validate_raw_state(observation_state, observation_components)
        if not self.state_history:
            self.state_history.append(fallback_state.copy())
        history = self._stack_history(fallback_state)
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
                "state": history.tolist(),
            },
            "robot_type": robot_type,
            "return_chunk": True,
        }
        if observation_components:
            payload["observation"]["state_components"] = {
                str(key): np.asarray(value, dtype=np.float32).reshape(-1).tolist()
                for key, value in observation_components.items()
            }
        if task_name:
            payload["task"] = task_name

        response = self.client._post_json("/infer", payload)
        full_chunk = self._decode_action_chunk(response)
        executed_steps = min(self.execute_steps, int(full_chunk.shape[0]))
        executed_full_chunk = full_chunk[:executed_steps].copy()
        continuous_hands = executed_full_chunk[:, EXPECTED_SEMANTIC_BODY_DIM:EXPECTED_ACTION_DIM]
        self.continuous_hand_queue.extend(row.copy() for row in continuous_hands)
        # The shared semantic runtime consumes the first 38 RefPose values plus
        # two legacy binary hand slots. Hands are overwritten per control step
        # by apply_semantic_action using the checkpoint's continuous 7+7 output.
        executed_chunk = np.concatenate(
            [
                executed_full_chunk[:, :EXPECTED_SEMANTIC_BODY_DIM],
                np.zeros(
                    (
                        executed_steps,
                        EXPECTED_SEMANTIC_ACTION_DIM - EXPECTED_SEMANTIC_BODY_DIM,
                    ),
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )
        boundary_l2 = None
        if self.previous_executed_last_action is not None:
            boundary_l2 = float(np.linalg.norm(full_chunk[0] - self.previous_executed_last_action))
        self.previous_executed_last_action = executed_full_chunk[-1].copy()

        trace: dict[str, Any] = {
            "event": "infer_chunk_aligned",
            "timestamp": time.time(),
            "step_idx": int(self.infer_index),
            "robot_type": robot_type,
            "task": task_name,
            "front_rgb_shape": list(rgb.shape),
            "observation_state_shape": list(history.shape),
            "observation_state_first": history[0].tolist(),
            "observation_state_last": history[-1].tolist(),
            "predicted_chunk_size": int(full_chunk.shape[0]),
            "executed_chunk_size": int(executed_steps),
            "predicted_action_dim": int(full_chunk.shape[1]),
            "runtime_action_dim": int(executed_chunk.shape[1]),
            "first_action": full_chunk[0].tolist(),
            "executed_last_action": executed_full_chunk[-1].tolist(),
            "runtime_executed_last_action": executed_chunk[-1].tolist(),
            "predicted_last_action": full_chunk[-1].tolist(),
            "boundary_l2": boundary_l2,
            "root_xy_norm_mean": float(np.linalg.norm(full_chunk[:, 0:2], axis=1).mean()),
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
        "[act_refpose] alignment enabled "
        f"input=raw64 history={alignment.history_steps}x@control_rate "
        f"output={EXPECTED_PREDICTED_CHUNK_STEPS}x{EXPECTED_ACTION_DIM} "
        f"body=semantic38 hands=continuous7+7 execute={alignment.execute_steps} "
        f"record_full_chunks={int(alignment.record_full_chunks)}"
    )
    return alignment
