from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SERVER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "serve_lerobot_vla_http.py"
SPEC = importlib.util.spec_from_file_location("serve_lerobot_vla_http", SERVER_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


def test_pi05_export_fields_are_removed_only_in_compat_copy(tmp_path: Path) -> None:
    policy_dir = tmp_path / "pretrained_model"
    policy_dir.mkdir()
    config = {
        "type": "pi05",
        "tokenizer_name": "/training/tokenizer",
        "adapt_action_head_from_pretrained": True,
        "chunk_size": 20,
    }
    config_path = policy_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (policy_dir / "model.safetensors").write_bytes(b"weights")

    temp_dir, compat_dir = SERVER._prepare_compat_policy_dir(policy_dir)
    try:
        compat_config = json.loads((compat_dir / "config.json").read_text(encoding="utf-8"))
        assert compat_config == {"type": "pi05", "chunk_size": 20}
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
        assert (compat_dir / "model.safetensors").resolve() == (
            policy_dir / "model.safetensors"
        ).resolve()
    finally:
        temp_dir.cleanup()


def test_pi05_action_head_export_field_requires_saved_weights(tmp_path: Path) -> None:
    policy_dir = tmp_path / "pretrained_model"
    policy_dir.mkdir()
    payload = {"type": "pi05", "adapt_action_head_from_pretrained": True}

    with pytest.raises(FileNotFoundError, match="exported weights"):
        SERVER._normalize_portable_pi05_config(payload, policy_dir)


def test_non_pi05_config_is_unchanged(tmp_path: Path) -> None:
    payload = {"type": "act", "tokenizer_name": "kept"}
    normalized, changed = SERVER._normalize_portable_pi05_config(payload, tmp_path)
    assert normalized == payload
    assert changed is False


def test_act_execution_steps_override_action_queue_length() -> None:
    config = SimpleNamespace(type="act", chunk_size=25, n_action_steps=25)

    SERVER._set_act_execution_steps(config, 5)

    assert config.n_action_steps == 5


@pytest.mark.parametrize("execution_steps", [0, 26])
def test_act_execution_steps_validate_chunk_range(execution_steps: int) -> None:
    config = SimpleNamespace(type="act", chunk_size=25, n_action_steps=25)

    with pytest.raises(ValueError, match="act-execution-steps"):
        SERVER._set_act_execution_steps(config, execution_steps)


def test_act_execution_steps_reject_non_act_policy() -> None:
    config = SimpleNamespace(type="stream", chunk_size=25, n_action_steps=25)

    with pytest.raises(ValueError, match="only supported for ACT"):
        SERVER._set_act_execution_steps(config, 5)
