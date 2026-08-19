from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from task_naming import TASK_RESULT_LABELS, episode_video_stem, result_label_for_task


def _load_server_module():
    server_path = Path(__file__).resolve().parent / "serve_lerobot_vla_http.py"
    spec = importlib.util.spec_from_file_location("serve_lerobot_vla_http", server_path)
    assert spec is not None and spec.loader is not None
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    return server


def _load_http_client_module():
    client_path = Path(__file__).resolve().parents[3] / "action_provider" / "lerobot_vla_http_client.py"
    spec = importlib.util.spec_from_file_location("lerobot_vla_http_client", client_path)
    assert spec is not None and spec.loader is not None
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    return client


@pytest.mark.parametrize(("task_name", "expected_label"), TASK_RESULT_LABELS.items())
def test_result_label_for_known_task(task_name: str, expected_label: str) -> None:
    assert result_label_for_task(task_name) == expected_label


def test_unknown_task_uses_the_actual_task_name() -> None:
    assert result_label_for_task("My New/Task") == "My_New_Task"


def test_episode_video_stem_starts_with_scene_label() -> None:
    assert episode_video_stem(
        "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
        "experiment__checkpoint-1000",
        seed=3,
        repeat_idx=2,
        episode_index=17,
    ) == "HSI_open_door__experiment__checkpoint-1000__seed_3__repeat_2__episode_17"


def test_server_mapped_mode_resolves_every_standard_task() -> None:
    server = _load_server_module()

    for task_name, expected_task_label in TASK_RESULT_LABELS.items():
        mapped_task_label, instruction = server._resolve_task_instruction(task_name)
        assert mapped_task_label == expected_task_label
        assert instruction == server.TASK_LANGUAGE_INSTRUCTIONS[expected_task_label]


def test_disable_stream_action_delta_refiner_bypasses_only_refinement() -> None:
    server = _load_server_module()
    config = SimpleNamespace(
        type="stream",
        action_delta_refiner_enabled=True,
        base_action_frequency_hz=20,
        action_delta_refiner_frequency_hz=50,
    )
    policy = SimpleNamespace(
        config=config,
        refine_action_step=lambda *_args, **_kwargs: "refined",
    )

    server._disable_stream_action_delta_refiner(policy)

    assert config.action_delta_refiner_enabled is True
    assert config.base_action_frequency_hz == 20
    assert config.action_delta_refiner_frequency_hz == 50
    assert policy.refine_action_step("base", "observation") == "base"
    assert policy._action_delta_refiner_runtime_disabled is True


def test_disable_action_delta_refiner_rejects_non_stream_policy() -> None:
    server = _load_server_module()
    policy = SimpleNamespace(config=SimpleNamespace(type="act"))

    with pytest.raises(ValueError, match="only supported for Stream"):
        server._disable_stream_action_delta_refiner(policy)


def test_server_splits_continuous_hand_actions_without_rescaling() -> None:
    server = _load_server_module()
    state = server.LeRobotServerState.__new__(server.LeRobotServerState)
    state.hand_action_keys = ["action.left_hand", "action.right_hand"]
    state.hand_action_key_dims = {"action.left_hand": 7, "action.right_hand": 7}
    raw = torch.arange(14, dtype=torch.float32).reshape(1, 14)

    split = state._split_hand_action(raw, source="test")

    np.testing.assert_array_equal(split["action.left_hand"], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(split["action.right_hand"], np.arange(7, 14, dtype=np.float32))


def test_server_splits_hand_action_chunks_along_the_last_dimension() -> None:
    server = _load_server_module()
    state = server.LeRobotServerState.__new__(server.LeRobotServerState)
    state.hand_action_keys = ["action.left_hand", "action.right_hand"]
    state.hand_action_key_dims = {"action.left_hand": 7, "action.right_hand": 7}
    raw = torch.arange(28, dtype=torch.float32).reshape(1, 2, 14)

    split = state._split_hand_action(raw, source="test")

    assert split["action.left_hand"].shape == (2, 7)
    assert split["action.right_hand"].shape == (2, 7)


def test_http_client_parses_optional_hand_payloads_and_rejects_bad_rank() -> None:
    client_module = _load_http_client_module()
    response = {
        "hand_actions": {
            "action.left_hand": [0.0] * 7,
            "action.right_hand": [1.0] * 7,
        }
    }

    parsed = client_module.LeRobotVLAHttpClient._parse_hand_actions(
        response,
        "hand_actions",
        expected_ndim=1,
    )

    assert set(parsed) == {"action.left_hand", "action.right_hand"}
    with pytest.raises(RuntimeError, match="rank 2"):
        client_module.LeRobotVLAHttpClient._parse_hand_actions(
            response,
            "hand_actions",
            expected_ndim=2,
        )
