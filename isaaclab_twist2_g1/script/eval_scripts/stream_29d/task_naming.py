#!/usr/bin/env python3

"""Canonical task and checkpoint labels for HumanoidArena evaluation artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TASK_RESULT_LABELS = {
    "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody": "HOI_double_desk",
    "Isaac-Move-Football-Single-G129-Dex3-Wholebody": "HOI_football",
    # Keep both spellings because the current task registration contains the
    # historical "Wholedoby" typo.
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby": "HOI_pp_box",
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody": "HOI_pp_box",
    "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody": "HSI_vision_navi",
    "Isaac-Move-Open-Door-G129-Dex3-Wholebody": "HSI_open_door",
    "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody": "HSI_sit_sofa",
    "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody": "HSI_boxing",
}


_RUN_TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(20\d{6}_\d{6})(?!\d)")


def result_label_for_task(task_name: str) -> str:
    normalized_task_name = str(task_name).strip()
    if not normalized_task_name:
        raise ValueError("task_name must not be empty")

    known_label = TASK_RESULT_LABELS.get(normalized_task_name)
    if known_label is not None:
        return known_label

    # Unknown/new tasks still get a deterministic label tied to the task that
    # was actually passed to the evaluator instead of falling back to a model
    # training-task name.
    sanitized = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in normalized_task_name
    ).strip("_")
    if not sanitized:
        raise ValueError(f"task_name has no usable result-label characters: {task_name!r}")
    return sanitized


def model_label_for_task_checkpoint(task_name: str, model_path: str | Path) -> str:
    """Build the shared per-episode artifact label for one task/checkpoint.

    A standard checkpoint path is either ``.../checkpoints/070000`` or
    ``.../checkpoints/070000/pretrained_model``.  Multi-task training run
    names do not identify the task being evaluated, so use the actual
    evaluation task label followed by the training run timestamp and
    checkpoint step, matching task-specific run labels such as
    ``HSI_sit_sofa_20260816_112409__200000``.
    """
    resolved_path = Path(model_path).expanduser().resolve()
    checkpoint_dir = resolved_path.parent if resolved_path.name == "pretrained_model" else resolved_path

    if checkpoint_dir.parent.name == "checkpoints":
        experiment_dir = checkpoint_dir.parent.parent
    else:
        # Retain a deterministic and informative fallback for non-standard
        # checkpoint layouts.
        experiment_dir = checkpoint_dir.parent

    task_label = result_label_for_task(task_name)
    checkpoint_label = result_label_for_task(checkpoint_dir.name)
    experiment_label = result_label_for_task(experiment_dir.name)
    timestamp_matches = _RUN_TIMESTAMP_PATTERN.findall(experiment_dir.name)
    if timestamp_matches:
        return f"{task_label}_{timestamp_matches[-1]}__{checkpoint_label}"
    return f"{task_label}__{experiment_label}__{checkpoint_label}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve the result label for an evaluation task")
    parser.add_argument("task_name")
    args = parser.parse_args()
    print(result_label_for_task(args.task_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
