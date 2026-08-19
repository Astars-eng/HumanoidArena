#!/usr/bin/env python3

"""Canonical result-directory labels for HumanoidArena evaluation tasks."""

from __future__ import annotations

import argparse


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


def episode_video_stem(
    task_name: str,
    model_label: str,
    seed: int,
    repeat_idx: int,
    episode_index: int,
) -> str:
    """Build a video stem prefixed by the canonical evaluation-scene label."""

    normalized_model_label = str(model_label).strip()
    if not normalized_model_label:
        raise ValueError("model_label must not be empty")
    return (
        f"{result_label_for_task(task_name)}__{normalized_model_label}"
        f"__seed_{int(seed)}__repeat_{int(repeat_idx)}__episode_{int(episode_index)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve the result label for an evaluation task")
    parser.add_argument("task_name")
    args = parser.parse_args()
    print(result_label_for_task(args.task_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
