#!/usr/bin/env python3
"""Fail-fast validation for Flow Matching SONIC motion-token raw107 checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXPECTED_STATE_KEYS = [
    "observation.joint_pos",
    "observation.joint_vel",
    "observation.ang_vel_b",
    "observation.gravity",
]
EXPECTED_STATE_SHAPES = {
    "observation.joint_pos": [29],
    "observation.joint_vel": [29],
    "observation.ang_vel_b": [3],
    "observation.gravity": [3],
    "observation.state": [64],
    "observation.images.front": [3, 224, 224],
}
EXPECTED_ACTION_KEYS = [
    "action.applied_action",
    "action.motion_token",
    "action.left_hand",
    "action.right_hand",
]
EXPECTED_ACTION_SHAPES = {
    "action.applied_action": [29],
    "action.motion_token": [64],
    "action.left_hand": [7],
    "action.right_hand": [7],
    "action": [107],
}


def _require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_checkpoint(path: str | Path) -> dict:
    policy_dir = Path(path).expanduser().resolve()
    if (policy_dir / "pretrained_model").is_dir():
        policy_dir = policy_dir / "pretrained_model"
    config_path = policy_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Flow Matching raw107 checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    _require_equal("policy type", config.get("type"), "flow_matching")
    _require_equal("n_obs_steps", config.get("n_obs_steps"), 10)
    _require_equal("obs_delta_sequence", config.get("obs_delta_sequence"), list(range(9, -1, -1)))
    _require_equal("chunk_size", config.get("chunk_size"), 25)
    _require_equal("n_action_steps", config.get("n_action_steps"), 25)
    _require_equal("num_inference_steps", config.get("num_inference_steps"), 10)
    _require_equal("integration_method", config.get("integration_method"), "midpoint")
    _require_equal("sigma_min", config.get("sigma_min"), 0.0)
    _require_equal("state_keys", config.get("state_keys"), EXPECTED_STATE_KEYS)
    _require_equal("action_keys", config.get("action_keys"), EXPECTED_ACTION_KEYS)

    input_features = config.get("input_features") or {}
    output_features = config.get("output_features") or {}
    for key, shape in EXPECTED_STATE_SHAPES.items():
        _require_equal(f"input feature {key}", (input_features.get(key) or {}).get("shape"), shape)
    for key, shape in EXPECTED_ACTION_SHAPES.items():
        _require_equal(f"output feature {key}", (output_features.get(key) or {}).get("shape"), shape)
    return config


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} /path/to/checkpoint[/pretrained_model]"
        )
    config = validate_checkpoint(sys.argv[1])
    print(
        "[fm_new107] checkpoint contract OK: "
        f"type={config['type']} state=29+29+3+3=64 history={config['n_obs_steps']} "
        f"image=224x224 action=29+motion_token64+7+7=107 "
        f"chunk={config['chunk_size']} execute={config['n_action_steps']} "
        f"integration={config['integration_method']}x{config['num_inference_steps']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
