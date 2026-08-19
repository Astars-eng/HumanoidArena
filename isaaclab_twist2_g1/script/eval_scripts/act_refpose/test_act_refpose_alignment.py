from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from act_refpose_alignment import configure_act_refpose_alignment


class _Client:
    def __init__(self, action_chunk: np.ndarray) -> None:
        self.action_chunk = action_chunk
        self.payloads = []
        self.traces = []
        self._trace_step_idx = 0

    def _post_json(self, path, payload):
        self.payloads.append((path, payload))
        return {"action_chunk": self.action_chunk.tolist()}

    def _append_trace(self, trace):
        self.traces.append(trace)

    def infer_chunk(self, *args, **kwargs):
        raise AssertionError("provider fetch hook must bypass client.infer_chunk")

    def infer_single(self, *args, **kwargs):
        raise AssertionError("provider fetch hook must bypass raw64 infer_single routing")


class _Provider:
    def __init__(self, action_chunk: np.ndarray) -> None:
        self._lerobot_http_client = _Client(action_chunk)
        self._vla_action_format = "semantic_v3"
        self._left_hand_target = np.zeros(7, dtype=np.float32)
        self._right_hand_target = np.zeros(7, dtype=np.float32)
        self.applied_semantic_actions = []
        self._lerobot_robot_type = "unitree_g1_refpose_v3_1"
        self.task_name = "task"

    def get_action(self, env):
        return env

    def on_env_reset(self):
        return None

    def _build_lerobot_raw107_observation(self):
        components = {
            "observation.joint_pos": np.arange(29, dtype=np.float32),
            "observation.joint_vel": np.arange(29, dtype=np.float32) + 100,
            "observation.ang_vel_b": np.arange(3, dtype=np.float32) + 200,
            "observation.gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        return np.concatenate(list(components.values())), components

    def _get_front_camera_rgb_for_vla(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def _fetch_lerobot_action_chunk(self):
        raise AssertionError("alignment hook was not installed")

    def _apply_lerobot_semantic_action(self, action):
        self.applied_semantic_actions.append(np.asarray(action).copy())
        self._left_hand_target.fill(-1.0)
        self._right_hand_target.fill(-1.0)


def test_refpose52_is_adapted_to_semantic_body_and_continuous_hands():
    full_chunk = np.arange(25 * 52, dtype=np.float32).reshape(25, 52)
    provider = _Provider(full_chunk)
    alignment = configure_act_refpose_alignment(
        provider,
        history_steps=10,
        execute_steps=5,
        record_full_chunks=True,
    )

    provider.get_action("env")
    runtime_chunk = provider._fetch_lerobot_action_chunk()

    assert runtime_chunk.shape == (5, 40)
    np.testing.assert_array_equal(runtime_chunk[:, :38], full_chunk[:5, :38])
    np.testing.assert_array_equal(runtime_chunk[:, 38:40], 0.0)
    payload_state = np.asarray(provider._lerobot_http_client.payloads[0][1]["observation"]["state"])
    assert payload_state.shape == (10, 64)

    provider._apply_lerobot_semantic_action(runtime_chunk[0])
    np.testing.assert_array_equal(provider._left_hand_target, full_chunk[0, 38:45])
    np.testing.assert_array_equal(provider._right_hand_target, full_chunk[0, 45:52])
    assert len(alignment.continuous_hand_queue) == 4


def test_reset_clears_refpose_history_and_hand_queue():
    provider = _Provider(np.zeros((25, 52), dtype=np.float32))
    alignment = configure_act_refpose_alignment(
        provider,
        history_steps=10,
        execute_steps=5,
        record_full_chunks=False,
    )
    provider.get_action(None)
    alignment.continuous_hand_queue.append(np.zeros(14, dtype=np.float32))
    provider.on_env_reset()
    assert not alignment.state_history
    assert not alignment.continuous_hand_queue
