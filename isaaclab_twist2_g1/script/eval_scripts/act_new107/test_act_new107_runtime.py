from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from act_new107_runtime import configure_act_new107_smoothing, ema_smooth_target
import eval_vla_suite


class _Provider:
    def __init__(self) -> None:
        self._vla_action_format = "raw107"
        self._raw107_body_source = "direct_raw"
        self._sonic_idx = np.arange(29)
        robot_data = SimpleNamespace(joint_pos=np.zeros((1, 29), dtype=np.float32))
        self.env = SimpleNamespace(scene={"robot": SimpleNamespace(data=robot_data)})
        self.target = np.ones(29, dtype=np.float32)
        self._latest_decoder_target = np.zeros(29, dtype=np.float32)

    def on_env_reset(self):
        return None

    def _run_gear_sonic_raw107_from_vla(self):
        return self.target.copy()


def test_direct_raw_ema_starts_from_current_joint_position():
    provider = _Provider()
    configure_act_new107_smoothing(provider, smooth_alpha=0.2)

    first = provider._run_gear_sonic_raw107_from_vla()
    second = provider._run_gear_sonic_raw107_from_vla()

    np.testing.assert_allclose(first, 0.2)
    np.testing.assert_allclose(second, 0.36)
    np.testing.assert_array_equal(provider._latest_decoder_target, second)


def test_reset_clears_smoothing_state():
    provider = _Provider()
    smoother = configure_act_new107_smoothing(provider, smooth_alpha=0.2)
    provider._run_gear_sonic_raw107_from_vla()

    provider.on_env_reset()

    assert smoother.previous_smoothed_target is None


def test_smoothing_requires_direct_raw():
    provider = _Provider()
    provider._raw107_body_source = "native_decoder"
    with pytest.raises(RuntimeError, match="direct_raw"):
        configure_act_new107_smoothing(provider, smooth_alpha=0.2)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, np.nan])
def test_ema_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        ema_smooth_target(np.zeros(29), np.ones(29), alpha)


def test_start_server_forwards_act_execution_steps(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(eval_vla_suite.subprocess, "Popen", fake_popen)
    args = SimpleNamespace(
        server_python=sys.executable,
        server_script=str(tmp_path / "serve.py"),
        server_device="cpu",
        server_host="127.0.0.1",
        server_port=18443,
        server_lerobot_src="",
        server_checkpoint_ref_remap=[],
        server_verbatim_task=False,
        server_stretch_image_to_policy_shape=False,
        server_scheme="http",
        tls_cert_file="",
        tls_key_file="",
        act_new107_execute_steps=7,
    )

    process, log_fp = eval_vla_suite._start_server(
        args,
        "/tmp/act-model",
        tmp_path / "server.log",
    )
    log_fp.close()

    assert process is not None
    cmd = captured["cmd"]
    option_index = cmd.index("--act-execution-steps")
    assert cmd[option_index + 1] == "7"
