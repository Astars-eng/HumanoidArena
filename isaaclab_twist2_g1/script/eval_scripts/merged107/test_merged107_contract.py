from __future__ import annotations

import json
from pathlib import Path

import pytest

from raw107_contract import EXPECTED_INPUT_SHAPES, EXPECTED_OUTPUT_SHAPES, validate_checkpoint


def _write_checkpoint(tmp_path: Path, family: str) -> Path:
    policy_dir = tmp_path / f"{family}_run" / "checkpoints" / "100000" / "pretrained_model"
    policy_dir.mkdir(parents=True)
    config = {
        "type": {"act": "act", "dp": "diffusion", "flowmatching": "multi_task_dit"}[family],
        "objective": "flow_matching" if family == "flowmatching" else None,
        "n_obs_steps": 1,
        "n_action_steps": 20,
        "input_features": {
            key: {"shape": shape, "type": "VISUAL" if "images" in key else "STATE"}
            for key, shape in EXPECTED_INPUT_SHAPES.items()
        },
        "output_features": {
            key: {"shape": shape, "type": "ACTION"}
            for key, shape in EXPECTED_OUTPUT_SHAPES.items()
        },
    }
    if family == "act":
        config["chunk_size"] = 20
    else:
        config["horizon"] = 24 if family == "dp" else 40
    preprocessor = {"steps": []}
    if family == "flowmatching":
        preprocessor["steps"].append(
            {"registry_name": "tokenizer_processor", "config": {"task_key": "task"}}
        )
    (policy_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (policy_dir / "policy_preprocessor.json").write_text(json.dumps(preprocessor), encoding="utf-8")
    (policy_dir / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    (policy_dir / "model.safetensors").write_bytes(b"test")
    return policy_dir


@pytest.mark.parametrize("family", ["act", "dp", "flowmatching"])
def test_accepts_all_merged_policy_families(tmp_path: Path, family: str) -> None:
    policy_dir = _write_checkpoint(tmp_path, family)
    assert validate_checkpoint(policy_dir)["_merged107_family"] == family
    assert validate_checkpoint(policy_dir.parent)["_merged107_family"] == family


def test_rejects_old_raw64_state_contract(tmp_path: Path) -> None:
    policy_dir = _write_checkpoint(tmp_path, "act")
    config_path = policy_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["input_features"]["observation.state"]["shape"] = [64]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="observation.state"):
        validate_checkpoint(policy_dir)


def test_flowmatching_requires_task_tokenizer(tmp_path: Path) -> None:
    policy_dir = _write_checkpoint(tmp_path, "flowmatching")
    (policy_dir / "policy_preprocessor.json").write_text('{"steps": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer_processor"):
        validate_checkpoint(policy_dir)
