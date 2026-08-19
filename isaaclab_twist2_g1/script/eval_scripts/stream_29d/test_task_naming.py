from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from task_naming import TASK_RESULT_LABELS, model_label_for_task_checkpoint, result_label_for_task


def _load_server_module():
    server_path = Path(__file__).resolve().parents[4] / "lerobot" / "scripts" / "serve_lerobot_vla_http.py"
    spec = importlib.util.spec_from_file_location("serve_lerobot_vla_http", server_path)
    assert spec is not None and spec.loader is not None
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    return server


@pytest.mark.parametrize(("task_name", "expected_label"), TASK_RESULT_LABELS.items())
def test_result_label_for_known_task(task_name: str, expected_label: str) -> None:
    assert result_label_for_task(task_name) == expected_label


def test_unknown_task_uses_the_actual_task_name() -> None:
    assert result_label_for_task("My New/Task") == "My_New_Task"


def test_multitask_checkpoint_model_label_starts_with_actual_eval_task() -> None:
    model_path = (
        "/tmp/outputs/finetune_stream_multisource_29d_20260815_153724/"
        "checkpoints/070000/pretrained_model"
    )
    assert (
        model_label_for_task_checkpoint(
            "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
            model_path,
        )
        == "HSI_open_door_20260815_153724__070000"
    )


def test_direct_checkpoint_path_uses_same_model_label() -> None:
    model_path = "/tmp/outputs/HSI_sit_sofa_20260816_112409/checkpoints/200000"
    assert (
        model_label_for_task_checkpoint(
            "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody",
            model_path,
        )
        == "HSI_sit_sofa_20260816_112409__200000"
    )


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
