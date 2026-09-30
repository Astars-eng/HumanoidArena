#!/usr/bin/env python3
"""Fail-fast validation for Stream SONIC DFS/MuJoCo raw29/raw43 checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXPECTED_STATE_KEYS = [
    "observation.joint_pos",
    "observation.joint_vel",
    "observation.ang_vel_b",
    "observation.gravity",
    "observation.last_action",
]
EXPECTED_STATE_SHAPES = {
    "observation.joint_pos": [29],
    "observation.joint_vel": [29],
    "observation.ang_vel_b": [3],
    "observation.gravity": [3],
    "observation.last_action": [29],
    "observation.state": [93],
    "observation.images.front": [3, 224, 224],
}
EXPECTED_ACTION_KEYS = ["action.applied_action"]
EXPECTED_ACTION_KEY_DIMS = {"action.applied_action": 29}


def _require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_checkpoint(path: str | Path, action_format: str | None = None) -> dict:
    policy_dir = Path(path).expanduser().resolve()
    if (policy_dir / "pretrained_model").is_dir():
        policy_dir = policy_dir / "pretrained_model"
    config_path = policy_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stream checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    policy_type = config.get("type")
    if policy_type != "stream":
        raise ValueError(f"policy type: expected 'stream', got {policy_type!r}")
    _require_equal("n_obs_steps", config.get("n_obs_steps"), 10)
    _require_equal("chunk_size", config.get("chunk_size"), 25)
    _require_equal("n_action_steps", config.get("n_action_steps"), 25)
    _require_equal("state_keys", config.get("state_keys"), EXPECTED_STATE_KEYS)
    action_keys = config.get("action_keys")
    if action_keys == EXPECTED_ACTION_KEYS:
        detected_format = "raw29"
        expected_dims = EXPECTED_ACTION_KEY_DIMS
    elif action_keys == ["action.applied_action", "action.left_hand", "action.right_hand"]:
        detected_format = "raw43"
        expected_dims = {"action.applied_action": 29, "action.left_hand": 7, "action.right_hand": 7}
    else:
        raise ValueError(f"action_keys: unsupported Stream action layout {action_keys!r}")
    if action_format is not None:
        _require_equal("action format", action_format, detected_format)

    input_features = config.get("input_features") or {}
    output_features = config.get("output_features") or {}
    for key in ("observation.state", "observation.images.front"):
        _require_equal(
            f"input feature {key}",
            (input_features.get(key) or {}).get("shape"),
            EXPECTED_STATE_SHAPES[key],
        )
    _require_equal(
        "output feature action",
        (output_features.get("action") or {}).get("shape"),
        [29] if detected_format == "raw29" else [43],
    )
    _require_equal("action_key_dims", config.get("action_key_dims"), expected_dims)
    return config


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} /path/to/checkpoint[/pretrained_model] [raw29|raw43]"
        )
    config = validate_checkpoint(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
    print(
        "[stream_29d] checkpoint contract OK: "
        f"type={config['type']} state=29+29+3+3+29=93 history={config['n_obs_steps']} "
        f"image=224x224 action={config['output_features']['action']['shape'][0]} "
        f"(DFS/MuJoCo body + optional Dex3 hands) chunk={config['chunk_size']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
