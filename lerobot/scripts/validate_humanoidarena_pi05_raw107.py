#!/usr/bin/env python
"""Fail-fast checks for HumanoidArena merged raw107 PI0.5 fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from safetensors import safe_open
from transformers import AutoTokenizer

EXPECTED_STATE_DIM = 93
EXPECTED_ACTION_DIM = 107
EXPECTED_IMAGE_SHAPE = [224, 224, 3]
REQUIRED_QUANTILES = {"q01", "q10", "q50", "q90", "q99"}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _shape(info: dict, key: str) -> list[int] | None:
    feature = info.get("features", {}).get(key, {})
    shape = feature.get("shape")
    return list(shape) if shape is not None else None


def validate_dataset(dataset_root: Path) -> None:
    info = _load_json(dataset_root / "meta" / "info.json")
    stats = _load_json(dataset_root / "meta" / "stats.json")

    expected_shapes = {
        "observation.state": [EXPECTED_STATE_DIM],
        "action": [EXPECTED_ACTION_DIM],
        "observation.images.front": EXPECTED_IMAGE_SHAPE,
    }
    for key, expected in expected_shapes.items():
        actual = _shape(info, key)
        if actual != expected:
            raise ValueError(f"{key} shape mismatch: expected={expected}, actual={actual}")

    for key in ("observation.state", "action"):
        available = set(stats.get(key, {}))
        missing = REQUIRED_QUANTILES - available
        if missing:
            raise ValueError(f"{key} is missing quantile stats: {sorted(missing)}")

    if not (dataset_root / "data").is_dir() or not (dataset_root / "videos").is_dir():
        raise FileNotFoundError(f"Dataset data/videos directories are incomplete: {dataset_root}")

    print(
        "[pi05-raw107] dataset OK: "
        f"state={EXPECTED_STATE_DIM} action={EXPECTED_ACTION_DIM} "
        f"episodes={info.get('total_episodes')} frames={info.get('total_frames')} "
        f"tasks={info.get('total_tasks')}"
    )


def validate_checkpoint(checkpoint_root: Path) -> None:
    config = _load_json(checkpoint_root / "config.json")
    if config.get("type") != "pi05":
        raise ValueError(f"Expected a pi05 checkpoint, got type={config.get('type')!r}")

    model_path = checkpoint_root / "model.safetensors"
    incomplete_path = checkpoint_root / "model.safetensors.incomplete"
    if not model_path.is_file():
        suffix = f"; download is still incomplete: {incomplete_path}" if incomplete_path.exists() else ""
        raise FileNotFoundError(f"Missing {model_path}{suffix}")

    required_shapes = {
        "action_in_proj.weight": (1024, 32),
        "action_out_proj.weight": (32, 1024),
        "action_out_proj.bias": (32,),
    }
    with safe_open(model_path, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        for key, expected in required_shapes.items():
            resolved_key = key if key in keys else f"model.{key}"
            if resolved_key not in keys:
                raise KeyError(f"Checkpoint is missing {key} (with or without model. prefix)")
            actual = tuple(tensors.get_slice(resolved_key).get_shape())
            if actual != expected:
                raise ValueError(f"{resolved_key} shape mismatch: expected={expected}, actual={actual}")
        tensor_count = len(keys)

    print(
        "[pi05-raw107] checkpoint OK: "
        f"base_action_dim={config.get('max_action_dim')} tensors={tensor_count} path={checkpoint_root}"
    )


def validate_tokenizer(dataset_root: Path, tokenizer_root: Path, max_length: int) -> None:
    required_files = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    missing = sorted(name for name in required_files if not (tokenizer_root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Tokenizer is incomplete at {tokenizer_root}; missing={missing}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet").index.astype(str).tolist()
    worst_length = 0
    worst_task = ""
    worst_bin = -1
    for bin_value in range(256):
        state = " ".join([str(bin_value)] * EXPECTED_STATE_DIM)
        for task in tasks:
            clean_task = task.strip().replace("_", " ").replace("\n", " ")
            prompt = f"Task: {clean_task}, State: {state};\nAction: "
            length = len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
            if length > worst_length:
                worst_length = length
                worst_task = task
                worst_bin = bin_value

    if worst_length > max_length:
        raise ValueError(
            f"tokenizer_max_length={max_length} would truncate a 93D PI0.5 state prompt; "
            f"worst_length={worst_length}, bin={worst_bin}, task={worst_task!r}"
        )
    print(
        "[pi05-raw107] tokenizer OK: "
        f"worst_prompt_tokens={worst_length} configured_max_length={max_length} path={tokenizer_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--tokenizer-max-length", type=int, default=512)
    parser.add_argument("--allow-incomplete-checkpoint", action="store_true")
    args = parser.parse_args()

    validate_dataset(args.dataset_root.resolve())
    validate_tokenizer(
        args.dataset_root.resolve(),
        args.tokenizer_root.resolve(),
        args.tokenizer_max_length,
    )
    try:
        validate_checkpoint(args.checkpoint_root.resolve())
    except FileNotFoundError:
        if not args.allow_incomplete_checkpoint:
            raise
        print("[pi05-raw107] checkpoint download is incomplete; dataset-only validation passed")


if __name__ == "__main__":
    main()
