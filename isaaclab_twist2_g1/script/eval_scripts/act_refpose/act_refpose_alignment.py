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


class ActRefposeAlignment:
    def __init__(
        self,
        action_provider: Any,
        *,
        history_steps: int,
        execute_steps: int,
        record_full_chunks: bool,
    ) -> None:
        if history_steps <= 0:
            raise ValueError(f"history_steps must be positive, got {history_steps}")
        if execute_steps <= 0:
            raise ValueError(f"execute_steps must be positive, got {execute_steps}")

        client = getattr(action_provider, "_lerobot_http_client", None)
        if client is None:
            raise RuntimeError("ACT ref-pose alignment requires the remote LeRobot HTTP client")
        if bool(getattr(action_provider, "_use_vla_raw107", False)):
            raise RuntimeError("ACT ref-pose alignment only supports the semantic_v3 40D action path")

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

        def aligned_infer_chunk(
            client_self,
            front_rgb: np.ndarray,
            observation_state: np.ndarray,
            robot_type: str,
            task: str | None = None,
            observation_components: dict[str, np.ndarray] | None = None,
        ) -> np.ndarray:
            return alignment.infer_chunk(
                front_rgb=front_rgb,
                observation_state=observation_state,
                robot_type=robot_type,
                task=task,
                observation_components=observation_components,
            )

        self.action_provider.get_action = MethodType(aligned_get_action, self.action_provider)
        self.action_provider.on_env_reset = MethodType(aligned_on_env_reset, self.action_provider)
        self.client.infer_chunk = MethodType(aligned_infer_chunk, self.client)
        setattr(self.action_provider, "_act_refpose_alignment", self)

    def reset_local_history(self) -> None:
        self.state_history.clear()
        self.previous_executed_last_action = None
        self.infer_index = 0

    def capture_current_state(self) -> None:
        state = np.asarray(
            self.action_provider._build_lerobot_vla_observation_state(),
            dtype=np.float32,
        ).reshape(-1)
        if state.size == 0 or not np.isfinite(state).all():
            raise ValueError("ACT ref-pose observation state is empty or contains NaN/Inf")
        self.state_history.append(state.copy())

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
        if chunk.ndim != 2:
            raise RuntimeError(f"Expected 2D action chunk, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise RuntimeError("ACT ref-pose action chunk contains NaN/Inf")
        return chunk

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
        history = self._stack_history(observation_state)
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
        executed_chunk = full_chunk[:executed_steps].copy()
        boundary_l2 = None
        if self.previous_executed_last_action is not None:
            boundary_l2 = float(np.linalg.norm(full_chunk[0] - self.previous_executed_last_action))
        self.previous_executed_last_action = executed_chunk[-1].copy()

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
            "first_action": full_chunk[0].tolist(),
            "executed_last_action": executed_chunk[-1].tolist(),
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
        f"history={alignment.history_steps}x state@control_rate "
        f"execute={alignment.execute_steps} predicted-chunk-prefix "
        f"record_full_chunks={int(alignment.record_full_chunks)}"
    )
    return alignment
