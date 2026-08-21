#!/usr/bin/env python3
"""Fail-fast contract validation for merged ACT/DP/flow-matching raw107 checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SUPPORTED_POLICY_TYPES = {"act", "diffusion", "multi_task_dit"}
EXPECTED_INPUT_SHAPES = {
    "observation.joint_pos": [29],
    "observation.joint_vel": [29],
    "observation.ang_vel_b": [3],
    "observation.gravity": [3],
    "observation.motion_token": [64],
    "observation.last_action": [29],
    "observation.images.front": [3, 224, 224],
    "observation.state": [93],
}
EXPECTED_OUTPUT_SHAPES = {
    "action.applied_action": [29],
    "action.motion_token": [64],
    "action.left_hand": [7],
    "action.right_hand": [7],
    "action": [107],
}


def _require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _require_positive_int(config: dict, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key}: expected a positive integer, got {value!r}")
    return value


def _resolve_policy_dir(path: str | Path) -> Path:
    policy_dir = Path(path).expanduser().resolve()
    if (policy_dir / "pretrained_model").is_dir():
        policy_dir = policy_dir / "pretrained_model"
    return policy_dir


def policy_family(config: dict) -> str:
    policy_type = config.get("type")
    if policy_type == "act":
        return "act"
    if policy_type == "diffusion":
        return "dp"
    if policy_type == "multi_task_dit" and config.get("objective") == "flow_matching":
        return "flowmatching"
    raise ValueError(
        "unsupported policy contract: expected ACT, diffusion, or "
        f"multi_task_dit flow_matching; got type={policy_type!r}, "
        f"objective={config.get('objective')!r}"
    )


def validate_checkpoint(path: str | Path) -> dict:
    policy_dir = _resolve_policy_dir(path)
    config_path = policy_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"merged107 checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    policy_type = config.get("type")
    if policy_type not in SUPPORTED_POLICY_TYPES:
        raise ValueError(
            f"policy type: expected one of {sorted(SUPPORTED_POLICY_TYPES)}, got {policy_type!r}"
        )
    family = policy_family(config)
    _require_positive_int(config, "n_obs_steps")
    _require_positive_int(config, "n_action_steps")
    if family == "act":
        _require_positive_int(config, "chunk_size")
    else:
        horizon = _require_positive_int(config, "horizon")
        if config["n_action_steps"] > horizon:
            raise ValueError(
                f"n_action_steps ({config['n_action_steps']}) must not exceed horizon ({horizon})"
            )

    input_features = config.get("input_features") or {}
    output_features = config.get("output_features") or {}
    for key, shape in EXPECTED_INPUT_SHAPES.items():
        _require_equal(f"input feature {key}", (input_features.get(key) or {}).get("shape"), shape)
    for key, shape in EXPECTED_OUTPUT_SHAPES.items():
        _require_equal(f"output feature {key}", (output_features.get(key) or {}).get("shape"), shape)

    for required_file in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (policy_dir / required_file).is_file():
            raise FileNotFoundError(f"checkpoint artifact not found: {policy_dir / required_file}")

    if family == "flowmatching":
        preprocessor = json.loads((policy_dir / "policy_preprocessor.json").read_text(encoding="utf-8"))
        tokenizer_steps = [
            step for step in preprocessor.get("steps", [])
            if step.get("registry_name") == "tokenizer_processor"
        ]
        if not tokenizer_steps:
            raise ValueError("flow-matching checkpoint preprocessor has no tokenizer_processor")
        _require_equal(
            "flow-matching tokenizer task_key",
            (tokenizer_steps[0].get("config") or {}).get("task_key"),
            "task",
        )

    validated = dict(config)
    validated["_merged107_family"] = family
    validated["_merged107_policy_dir"] = str(policy_dir)
    return validated


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} /path/to/checkpoint[/pretrained_model]"
        )
    config = validate_checkpoint(sys.argv[1])
    sequence_length = config.get("chunk_size") or config.get("horizon")
    print(
        "[merged107] checkpoint contract OK: "
        f"family={config['_merged107_family']} type={config['type']} "
        f"state=29+29+3+3+29=93 image=224x224 "
        f"action=29+64+7+7=107 sequence={sequence_length} "
        f"executed_steps={config['n_action_steps']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
