from __future__ import annotations

import unittest

import numpy as np

from act_refpose_alignment import (
    EXPECTED_ACTION_DIM,
    EXPECTED_HISTORY_STEPS,
    EXPECTED_PREDICTED_CHUNK_STEPS,
    ActRefposeAlignment,
)
from act_refpose_image import resize_rgb_to_policy_shape


class _FakeClient:
    def __init__(self) -> None:
        self.payloads = []
        self.traces = []
        self._trace_step_idx = 0

    def _post_json(self, path, payload):
        if path != "/infer":
            raise AssertionError(path)
        self.payloads.append(payload)
        chunk = np.arange(
            EXPECTED_PREDICTED_CHUNK_STEPS * EXPECTED_ACTION_DIM,
            dtype=np.float32,
        ).reshape(EXPECTED_PREDICTED_CHUNK_STEPS, EXPECTED_ACTION_DIM)
        return {"action_chunk": chunk.tolist()}

    def _append_trace(self, payload):
        self.traces.append(payload)


class _FakeProvider:
    def __init__(self) -> None:
        self._lerobot_http_client = _FakeClient()
        self._vla_action_format = "semantic_v3"
        self._lerobot_robot_type = "unitree_g1_refpose_v3_1"
        self.task_name = "HOI_double_desk"
        self.marker = 0.0

    def get_action(self, env):
        return None

    def on_env_reset(self):
        return "reset"

    def _fetch_lerobot_action_chunk(self):
        raise AssertionError("alignment hook was not installed")

    def _build_lerobot_raw107_observation(self):
        components = {
            "observation.joint_pos": np.full(29, self.marker, dtype=np.float32),
            "observation.joint_vel": np.full(29, self.marker + 100, dtype=np.float32),
            "observation.ang_vel_b": np.full(3, self.marker + 200, dtype=np.float32),
            "observation.gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        state = np.concatenate(list(components.values())).astype(np.float32)
        return state, components

    def _get_front_camera_rgb_for_vla(self):
        return np.zeros((8, 12, 3), dtype=np.uint8)


class ActRefposeAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = _FakeProvider()
        self.alignment = ActRefposeAlignment(
            self.provider,
            history_steps=EXPECTED_HISTORY_STEPS,
            execute_steps=5,
            record_full_chunks=True,
        )

    def test_history_is_oldest_to_current_and_clipped_to_t_minus_9(self):
        for marker in range(12):
            self.provider.marker = float(marker)
            self.alignment.capture_current_state()

        history = self.alignment._stack_history(
            self.alignment._read_current_raw_state()
        )
        self.assertEqual(history.shape, (10, 64))
        self.assertEqual(float(history[0, 0]), 2.0)
        self.assertEqual(float(history[-1, 0]), 11.0)
        np.testing.assert_array_equal(
            history[-1, 58:61], np.full(3, 211.0, dtype=np.float32)
        )
        np.testing.assert_array_equal(history[-1, 61:64], np.array([0.0, 0.0, -1.0]))

    def test_episode_start_left_pads_earliest_state(self):
        self.provider.marker = 7.0
        self.alignment.capture_current_state()
        history = self.alignment._stack_history(
            self.alignment._read_current_raw_state()
        )
        np.testing.assert_array_equal(history, np.repeat(history[-1:], 10, axis=0))

    def test_fetch_sends_10x64_and_returns_only_execution_prefix(self):
        for marker in range(10):
            self.provider.marker = float(marker)
            self.alignment.capture_current_state()

        executed = self.provider._fetch_lerobot_action_chunk()
        self.assertEqual(executed.shape, (5, 40))
        payload = self.provider._lerobot_http_client.payloads[-1]
        self.assertEqual(np.asarray(payload["observation"]["state"]).shape, (10, 64))
        self.assertTrue(payload["return_chunk"])
        self.assertEqual(payload["robot_type"], "unitree_g1_refpose_v3_1")
        self.assertEqual(payload["task"], "HOI_double_desk")
        trace = self.provider._lerobot_http_client.traces[-1]
        self.assertEqual(trace["predicted_chunk_size"], 25)
        self.assertEqual(trace["executed_chunk_size"], 5)
        self.assertEqual(np.asarray(trace["action_chunk"]).shape, (25, 40))

    def test_checkpoint_action_horizon_executes_full_chunk(self):
        provider = _FakeProvider()
        alignment = ActRefposeAlignment(
            provider,
            history_steps=EXPECTED_HISTORY_STEPS,
            execute_steps=EXPECTED_PREDICTED_CHUNK_STEPS,
            record_full_chunks=False,
        )
        alignment.capture_current_state()

        executed = provider._fetch_lerobot_action_chunk()

        self.assertEqual(
            executed.shape,
            (EXPECTED_PREDICTED_CHUNK_STEPS, EXPECTED_ACTION_DIM),
        )
        self.assertEqual(
            provider._lerobot_http_client.traces[-1]["executed_chunk_size"],
            EXPECTED_PREDICTED_CHUNK_STEPS,
        )

    def test_reset_clears_history_and_boundary_state(self):
        self.alignment.capture_current_state()
        self.alignment.previous_executed_last_action = np.ones(40, dtype=np.float32)
        self.assertEqual(self.provider.on_env_reset(), "reset")
        self.assertEqual(len(self.alignment.state_history), 0)
        self.assertIsNone(self.alignment.previous_executed_last_action)

    def test_rejects_wrong_history_length(self):
        provider = _FakeProvider()
        with self.assertRaisesRegex(ValueError, "requires history_steps=10"):
            ActRefposeAlignment(
                provider,
                history_steps=9,
                execute_steps=5,
                record_full_chunks=False,
            )

    def test_front_image_matches_training_resize_contract(self):
        image = np.arange(480 * 640 * 4, dtype=np.uint8).reshape(480, 640, 4)

        resized = resize_rgb_to_policy_shape(image, (3, 224, 224))

        self.assertEqual(resized.shape, (224, 224, 3))
        self.assertEqual(resized.dtype, np.uint8)
        self.assertTrue(resized.flags.c_contiguous)

    def test_front_image_rejects_non_chw_policy_contract(self):
        with self.assertRaisesRegex(ValueError, "3-channel CHW"):
            resize_rgb_to_policy_shape(
                np.zeros((224, 224, 3), dtype=np.uint8),
                (224, 224, 3),
            )


if __name__ == "__main__":
    unittest.main()
