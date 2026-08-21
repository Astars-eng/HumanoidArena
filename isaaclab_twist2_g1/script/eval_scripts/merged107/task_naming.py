#!/usr/bin/env python3
"""Task prompts and artifact labels for the seven-task merged107 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from raw107_contract import policy_family, validate_checkpoint


TASK_SPECS = {
    "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody": (
        "HOI_double_desk",
        "Put the hammer from the right table into the basket on the left table.",
    ),
    "Isaac-Move-Football-Single-G129-Dex3-Wholebody": (
        "HOI_football",
        "Kick the football into the goal.",
    ),
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby": (
        "HOI_pp_box",
        "Move the box from the table onto the shelf.",
    ),
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody": (
        "HOI_pp_box",
        "Move the box from the table onto the shelf.",
    ),
    "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody": (
        "HSI_boxing",
        "Strike the green markers on the punching bag.",
    ),
    "Isaac-Move-Open-Door-G129-Dex3-Wholebody": (
        "HSI_open_door",
        "Open the door.",
    ),
    "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody": (
        "HSI_sit_sofa",
        "Sit on the sofa.",
    ),
    "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody": (
        "HSI_vision_navi",
        "Avoid obstacles and move to the yellow marked area.",
    ),
}


def _task_spec(task_name: str) -> tuple[str, str]:
    normalized = str(task_name).strip()
    if not normalized:
        raise ValueError("task_name must not be empty")
    try:
        return TASK_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown merged107 task {normalized!r}; expected one of {sorted(TASK_SPECS)}"
        ) from exc


def result_label_for_task(task_name: str) -> str:
    return _task_spec(task_name)[0]


def policy_prompt_for_task(task_name: str) -> str:
    return _task_spec(task_name)[1]


def model_label_for_task_checkpoint(task_name: str, model_path: str | Path) -> str:
    resolved_path = Path(model_path).expanduser().resolve()
    checkpoint_dir = resolved_path.parent if resolved_path.name == "pretrained_model" else resolved_path
    experiment_dir = (
        checkpoint_dir.parent.parent
        if checkpoint_dir.parent.name == "checkpoints"
        else checkpoint_dir.parent
    )
    config = validate_checkpoint(resolved_path)
    return "__".join(
        (
            result_label_for_task(task_name),
            policy_family(config),
            experiment_dir.name,
            checkpoint_dir.name,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a merged107 task prompt or result label")
    parser.add_argument("mode", choices=("prompt", "label"))
    parser.add_argument("task_name")
    args = parser.parse_args()
    if args.mode == "prompt":
        print(policy_prompt_for_task(args.task_name))
    else:
        print(result_label_for_task(args.task_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
