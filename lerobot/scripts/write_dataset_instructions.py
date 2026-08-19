#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASET_TOKENS = ("merged", "sonic", "twist2")


def iter_target_datasets(task_root: Path, dataset_tokens: tuple[str, ...]) -> list[Path]:
    # Some released/downloaded datasets use a flat layout where task_root is
    # itself a complete LeRobot dataset, for example:
    #   HumanoidArena_lerobot_sonic/HSI_open_door/meta/tasks.parquet
    # Keep supporting the original nested layout below as well.
    if (task_root / "meta" / "tasks.parquet").is_file():
        return [task_root]

    token_set = tuple(token.lower() for token in dataset_tokens)
    dataset_dirs: list[Path] = []
    for child in sorted(task_root.iterdir()):
        if not child.is_dir():
            continue
        if not any(token in child.name.lower() for token in token_set):
            continue
        tasks_path = child / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            continue
        dataset_dirs.append(child)
    return dataset_dirs


def write_instruction(tasks_path: Path, instruction: str) -> None:
    df = pd.DataFrame({"task_index": [0]}, index=pd.Index([instruction], name="task"))
    df.to_parquet(tasks_path)


def update_episode_instructions(dataset_dir: Path, instruction: str) -> list[Path]:
    """Keep LeRobot v3 per-episode task text consistent with tasks.parquet."""
    updated: list[Path] = []
    for episodes_path in sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(episodes_path)
        if "tasks" not in df.columns:
            continue
        df["tasks"] = pd.Series(
            [np.asarray([instruction], dtype=object) for _ in range(len(df))],
            index=df.index,
            dtype=object,
        )
        df.to_parquet(episodes_path, index=False)
        updated.append(episodes_path)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline writer for LeRobot task instructions across merged/sonic/twist2 datasets."
    )
    parser.add_argument(
        "task_root",
        type=Path,
        help=(
            "Task root directory. It may be a LeRobot dataset itself or a parent "
            "containing merged/sonic/twist2 dataset directories."
        ),
    )
    parser.add_argument("instruction", type=str, help="English instruction to write into meta/tasks.parquet")
    parser.add_argument(
        "--dataset-tokens",
        nargs="+",
        default=list(DEFAULT_DATASET_TOKENS),
        help="Subdirectory name tokens to match. Defaults to: merged sonic twist2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which datasets would be updated without modifying files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_root = args.task_root.resolve()
    if not task_root.is_dir():
        raise FileNotFoundError(f"Task root does not exist or is not a directory: {task_root}")

    dataset_dirs = iter_target_datasets(task_root, tuple(args.dataset_tokens))
    if not dataset_dirs:
        print(f"No matching datasets found under {task_root}")
        return 1

    print(f"task_root: {task_root}")
    print(f"instruction: {args.instruction}")
    print(f"dataset_tokens: {tuple(args.dataset_tokens)}")

    for dataset_dir in dataset_dirs:
        tasks_path = dataset_dir / "meta" / "tasks.parquet"
        episode_paths = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if args.dry_run:
            print(f"DRY_RUN {tasks_path}")
            for episodes_path in episode_paths:
                print(f"DRY_RUN {episodes_path} (tasks column, if present)")
            continue
        write_instruction(tasks_path, args.instruction)
        print(f"UPDATED {tasks_path}")
        for episodes_path in update_episode_instructions(dataset_dir, args.instruction):
            print(f"UPDATED {episodes_path} (tasks column)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
