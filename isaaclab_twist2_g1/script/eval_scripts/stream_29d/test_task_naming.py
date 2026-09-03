from __future__ import annotations

import importlib.util
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval_vla_suite as eval_suite
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


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        (0.0, [1.0, 2.0]),
        (0.25, [1.5, 1.5]),
        (0.5, [2.0, 1.0]),
    ],
)
def test_action_delta_refiner_weight_scales_only_the_residual(weight, expected) -> None:
    server = _load_server_module()
    policy = SimpleNamespace(
        config=SimpleNamespace(type="stream", action_delta_refiner_enabled=True),
        refine_action_step=lambda _base, *_args, **_kwargs: server.torch.tensor([[3.0, 0.0]]),
    )

    server._set_stream_action_delta_refiner_weight(policy, weight)

    base = server.torch.tensor([[1.0, 2.0]])
    assert policy.refine_action_step(base, {})[0].tolist() == pytest.approx(expected)
    assert policy._action_delta_refiner_runtime_weight == pytest.approx(weight)


def test_action_delta_refiner_weight_one_preserves_original_callable() -> None:
    server = _load_server_module()
    original = lambda *_args, **_kwargs: "original"
    policy = SimpleNamespace(
        config=SimpleNamespace(type="stream", action_delta_refiner_enabled=True),
        refine_action_step=original,
    )

    server._set_stream_action_delta_refiner_weight(policy, 1.0)

    assert policy.refine_action_step is original


def test_action_delta_refiner_weight_rejects_negative_values() -> None:
    server = _load_server_module()
    policy = SimpleNamespace(
        config=SimpleNamespace(type="stream", action_delta_refiner_enabled=True),
        refine_action_step=lambda base, *_args, **_kwargs: base,
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        server._set_stream_action_delta_refiner_weight(policy, -0.1)


def test_stream_execution_horizon_overrides_n_action_steps() -> None:
    server = _load_server_module()
    config = SimpleNamespace(type="stream", chunk_size=25, n_action_steps=25)

    server._set_stream_execution_horizon(config, 3)

    assert config.n_action_steps == 3


@pytest.mark.parametrize("horizon", [0, 26])
def test_stream_execution_horizon_validates_checkpoint_chunk_size(horizon: int) -> None:
    server = _load_server_module()
    config = SimpleNamespace(type="stream", chunk_size=25, n_action_steps=25)

    with pytest.raises(ValueError, match="stream-execution-horizon"):
        server._set_stream_execution_horizon(config, horizon)


def test_stream_num_inference_steps_overrides_checkpoint_value() -> None:
    server = _load_server_module()
    config = SimpleNamespace(
        type="stream",
        num_inference_steps=20,
        attention_trace_enabled=False,
    )

    server._set_stream_num_inference_steps(config, 10)

    assert config.num_inference_steps == 10


@pytest.mark.parametrize("num_inference_steps", [0, -1, 1.5])
def test_stream_num_inference_steps_requires_positive_integer(num_inference_steps: object) -> None:
    server = _load_server_module()
    config = SimpleNamespace(
        type="stream",
        num_inference_steps=10,
        attention_trace_enabled=False,
    )

    with pytest.raises(ValueError, match="num-inference-steps"):
        server._set_stream_num_inference_steps(config, num_inference_steps)


def test_stream_num_inference_steps_validates_attention_trace_step() -> None:
    server = _load_server_module()
    config = SimpleNamespace(
        type="stream",
        num_inference_steps=10,
        attention_trace_enabled=True,
        attention_trace_flow_step=5,
    )

    with pytest.raises(ValueError, match="attention_trace_flow_step"):
        server._set_stream_num_inference_steps(config, 5)


def test_default_num_inference_steps_is_a_noop_for_non_stream_policy() -> None:
    server = _load_server_module()
    config = SimpleNamespace(type="act")

    server._set_stream_num_inference_steps(config, 10)

    assert not hasattr(config, "num_inference_steps")


def test_custom_num_inference_steps_rejects_non_stream_policy() -> None:
    server = _load_server_module()
    config = SimpleNamespace(type="act")

    with pytest.raises(ValueError, match="only supported for Stream"):
        server._set_stream_num_inference_steps(config, 20)


def test_start_server_forwards_stream_runtime_overrides(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(eval_suite.subprocess, "Popen", fake_popen)
    args = SimpleNamespace(
        server_python="/tmp/fake-python",
        server_script="/tmp/fake-server.py",
        server_device="cpu",
        server_host="127.0.0.1",
        server_port=18443,
        server_lerobot_src="",
        server_checkpoint_ref_remap=[],
        server_verbatim_task=False,
        server_stretch_image_to_policy_shape=False,
        server_disable_action_delta_refiner=False,
        server_stream_execution_horizon=3,
        server_action_delta_refiner_weight=0.25,
        server_num_inference_steps=12,
        server_zero_inference_noise=False,
        server_scheme="http",
        tls_cert_file="",
        tls_key_file="",
    )

    process, log_fp = eval_suite._start_server(args, "/tmp/model", tmp_path / "server.log")
    log_fp.close()

    assert process is not None
    cmd = captured["cmd"]
    horizon_index = cmd.index("--stream-execution-horizon")
    inference_steps_index = cmd.index("--num-inference-steps")
    weight_index = cmd.index("--action-delta-refiner-weight")
    assert cmd[horizon_index + 1] == "3"
    assert cmd[inference_steps_index + 1] == "12"
    assert cmd[weight_index + 1] == "0.25"


def test_server_infer_reports_base_refined_delta_and_chunk_metadata() -> None:
    server = _load_server_module()
    state = object.__new__(server.LeRobotServerState)
    state.lock = threading.Lock()
    state._amp_context = nullcontext
    state._prepare_observation = lambda *_args: {"observation.state": "prepared"}

    class FakePolicy:
        def select_action(self, _observation):
            return server.torch.tensor([[0.3, 0.8]])

        def get_last_action_before_refiner(self):
            return server.torch.tensor([[0.1, 0.5]])

        def get_last_action_metadata(self):
            return {"chunk_index": 7, "chunk_step_index": 0, "chunk_start": True}

    state.policy = FakePolicy()
    state.postprocessor = lambda action: action * 10.0
    state.config = SimpleNamespace(type="stream", n_action_steps=3, num_inference_steps=12)
    state.effective_action_delta_refiner_weight = 0.5

    action, trace = state.infer({}, "g1", "HOI_football")

    assert action.reshape(-1).tolist() == pytest.approx([3.0, 8.0])
    assert trace["base_action"] == pytest.approx([1.0, 5.0])
    assert trace["refined_action"] == pytest.approx([3.0, 8.0])
    assert trace["refiner_applied_delta"] == pytest.approx([2.0, 3.0])
    assert trace["normalized_refiner_applied_delta"] == pytest.approx([0.2, 0.3])
    assert trace["action_metadata"]["chunk_index"] == 7
    assert trace["action_metadata"]["execution_horizon"] == 3
    assert trace["action_metadata"]["num_inference_steps"] == 12
    assert trace["action_metadata"]["action_delta_refiner_weight"] == pytest.approx(0.5)
