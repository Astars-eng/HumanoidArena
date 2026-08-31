#!/usr/bin/env python3
"""Plot representative lower-body raw-action traces from ACT and Stream runs.

The ACT raw107 server logs contain the returned 107-D action at every control
step.  Its first 29 values and Stream's ``action.applied_action`` use the same
DFS/MuJoCo joint order.  This script selects episodes closest to the median
terminal step count in each requested outcome class, then plots the 12 leg and
3 waist components at 50 Hz.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONTROL_HZ = 50.0
BODY_JOINTS_DFS = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
LEFT_LEG = tuple(range(0, 6))
RIGHT_LEG = tuple(range(6, 12))
WAIST = tuple(range(12, 15))
LEG_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
WAIST_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
RESET_RE = re.compile(r"reset count=\d+ seed=(None|\d+)")
ACTION_RE = re.compile(r"infer count=\d+ action=(\[.*\])$")
SEED_LOG_RE = re.compile(r"__seed_(\d+)__batch\.log$")


@dataclass
class EpisodeTrace:
    source: str
    outcome: str
    episode_index: int
    seed: int
    repeat_idx: int
    episode_seed: int
    episode_steps: int
    actions: np.ndarray
    source_trace: Path
    video_path: Path

    @property
    def time(self) -> np.ndarray:
        return np.arange(len(self.actions), dtype=np.float64) / CONTROL_HZ

    @property
    def terminal_time(self) -> float:
        return self.episode_steps / CONTROL_HZ

    @property
    def label(self) -> str:
        return f"{self.source} {self.outcome} | episode {self.episode_index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-run", type=Path, required=True)
    parser.add_argument("--stream-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    return parser.parse_args()


def read_summary(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_near_median(rows: list[dict[str, str]], outcome: str, count: int) -> tuple[list[dict[str, str]], float]:
    eligible = [row for row in rows if row["failure_reason"] == outcome]
    if len(eligible) < count:
        raise ValueError(f"Need {count} {outcome} episodes, found {len(eligible)}")
    median_steps = float(np.median([int(row["episode_steps"]) for row in eligible]))
    eligible.sort(
        key=lambda row: (
            abs(int(row["episode_steps"]) - median_steps),
            int(row["episode_index"]),
        )
    )
    return eligible[:count], median_steps


def server_log_for_row(run_dir: Path, row: dict[str, str]) -> Path:
    batch_match = SEED_LOG_RE.search(Path(row["log_path"]).name)
    candidates: list[Path] = []
    if batch_match:
        candidates = sorted((run_dir / "logs").glob(f"server__*__task_{batch_match.group(1)}__*.log"))
    if not candidates:
        candidates = sorted((run_dir / "logs").glob("server__*.log"))
    target = f"seed={row['episode_seed']} "
    for candidate in candidates:
        with candidate.open(errors="replace") as handle:
            if any(target in line and "reset count=" in line for line in handle):
                return candidate
    raise FileNotFoundError(f"No ACT server log contains episode seed {row['episode_seed']}")


def load_act_actions(server_log: Path, episode_seed: int) -> np.ndarray:
    current_seed: int | None = None
    values: list[list[float]] = []
    found_target = False
    with server_log.open(errors="replace") as handle:
        for line in handle:
            reset_match = RESET_RE.search(line)
            if reset_match:
                parsed = reset_match.group(1)
                new_seed = None if parsed == "None" else int(parsed)
                if new_seed is not None and new_seed != current_seed:
                    if found_target:
                        break
                    current_seed = new_seed
                    found_target = current_seed == episode_seed
                continue
            if not found_target:
                continue
            action_match = ACTION_RE.search(line.rstrip())
            if action_match:
                action = ast.literal_eval(action_match.group(1))
                if len(action) < 29:
                    raise ValueError(f"ACT action has {len(action)} dimensions in {server_log}")
                values.append(action[:29])
    if not values:
        raise ValueError(f"No ACT actions found for episode seed {episode_seed} in {server_log}")
    return np.asarray(values, dtype=np.float32)


def load_stream_actions(trace_path: Path) -> np.ndarray:
    values: list[list[float]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") != "infer_single":
                continue
            action = record.get("action.applied_action", record.get("action"))
            if action is None or len(action) != 29:
                raise ValueError(f"Invalid Stream applied action in {trace_path}")
            values.append(action)
    if not values:
        raise ValueError(f"No Stream infer_single actions in {trace_path}")
    return np.asarray(values, dtype=np.float32)


def resolve_existing_path(path_text: str, run_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    candidate = run_dir / path.name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(path)


def make_act_trace(run_dir: Path, row: dict[str, str]) -> EpisodeTrace:
    server_log = server_log_for_row(run_dir, row)
    actions = load_act_actions(server_log, int(row["episode_seed"]))
    return EpisodeTrace(
        source="ACT raw107",
        outcome="success",
        episode_index=int(row["episode_index"]),
        seed=int(row["seed"]),
        repeat_idx=int(row["repeat_idx"]),
        episode_seed=int(row["episode_seed"]),
        episode_steps=int(row["episode_steps"]),
        actions=actions,
        source_trace=server_log.resolve(),
        video_path=Path(row["video_path"]).resolve(),
    )


def make_stream_trace(run_dir: Path, row: dict[str, str]) -> EpisodeTrace:
    trace_path = resolve_existing_path(row["vla_trace_path"], run_dir / "recordings" / "vla_outputs")
    actions = load_stream_actions(trace_path)
    return EpisodeTrace(
        source="Stream 29D",
        outcome="fall",
        episode_index=int(row["episode_index"]),
        seed=int(row["seed"]),
        repeat_idx=int(row["repeat_idx"]),
        episode_seed=int(row["episode_seed"]),
        episode_steps=int(row["episode_steps"]),
        actions=actions,
        source_trace=trace_path,
        video_path=Path(row["video_path"]).resolve(),
    )


def decorate_axis(ax: plt.Axes, trace: EpisodeTrace) -> None:
    outcome_color = "#2ca02c" if trace.outcome == "success" else "#d62728"
    ax.axvline(trace.terminal_time, color=outcome_color, linestyle="--", linewidth=1.4)
    if trace.time[-1] > trace.terminal_time:
        ax.axvspan(trace.terminal_time, trace.time[-1], color=outcome_color, alpha=0.07)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Raw action")


def plot_joint_group(
    ax: plt.Axes,
    trace: EpisodeTrace,
    indices: tuple[int, ...],
    colors: tuple[str, ...],
    *,
    legend: bool,
) -> None:
    for index, color in zip(indices, colors, strict=True):
        ax.plot(trace.time, trace.actions[:, index], label=BODY_JOINTS_DFS[index], color=color, linewidth=1.05)
    decorate_axis(ax, trace)
    if legend:
        ax.legend(fontsize=7, ncol=2, loc="best")


def plot_individual(trace: EpisodeTrace, output_dir: Path) -> list[Path]:
    stem = f"{trace.source.lower().replace(' ', '_')}__{trace.outcome}__episode_{trace.episode_index}"
    leg_path = output_dir / f"{stem}__legs.png"
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    plot_joint_group(axes[0], trace, LEFT_LEG, LEG_COLORS, legend=True)
    axes[0].set_title(f"{trace.label} — left-leg raw actions")
    plot_joint_group(axes[1], trace, RIGHT_LEG, LEG_COLORS, legend=True)
    axes[1].set_title(f"{trace.label} — right-leg raw actions")
    fig.savefig(leg_path, dpi=180)
    plt.close(fig)

    waist_path = output_dir / f"{stem}__waist.png"
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    plot_joint_group(ax, trace, WAIST, WAIST_COLORS, legend=True)
    ax.set_title(f"{trace.label} — waist raw actions")
    fig.savefig(waist_path, dpi=180)
    plt.close(fig)
    return [leg_path, waist_path]


def plot_combined(traces: list[EpisodeTrace], output_dir: Path) -> Path:
    combined_path = output_dir / "football_raw_actions__act_success_vs_stream_fall.png"
    fig, axes = plt.subplots(len(traces), 3, figsize=(19, 4.0 * len(traces)), constrained_layout=True)
    if len(traces) == 1:
        axes = np.asarray([axes])
    for row_idx, trace in enumerate(traces):
        plot_joint_group(axes[row_idx, 0], trace, LEFT_LEG, LEG_COLORS, legend=row_idx == 0)
        plot_joint_group(axes[row_idx, 1], trace, RIGHT_LEG, LEG_COLORS, legend=row_idx == 0)
        plot_joint_group(axes[row_idx, 2], trace, WAIST, WAIST_COLORS, legend=row_idx == 0)
        prefix = (
            f"{trace.source} {trace.outcome} | ep {trace.episode_index} | "
            f"seed {trace.seed}/repeat {trace.repeat_idx} | terminal {trace.terminal_time:.2f}s"
        )
        axes[row_idx, 0].set_title(prefix + "\nLeft leg")
        axes[row_idx, 1].set_title("Right leg")
        axes[row_idx, 2].set_title("Waist")
    fig.suptitle(
        "Football lower-body raw actions (DFS/MuJoCo order, 50 Hz)\n"
        "Dashed line = success/fall decision; shaded tail = 10 post-termination recording steps",
        fontsize=15,
    )
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)
    return combined_path


def write_trace_csv(traces: list[EpisodeTrace], output_dir: Path) -> Path:
    path = output_dir / "selected_raw_action_traces.csv"
    fieldnames = [
        "source",
        "outcome",
        "episode_index",
        "seed",
        "repeat_idx",
        "episode_seed",
        "control_step",
        "time_sec",
        "terminal_step",
        *BODY_JOINTS_DFS[:15],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trace in traces:
            for step, action in enumerate(trace.actions):
                row = {
                    "source": trace.source,
                    "outcome": trace.outcome,
                    "episode_index": trace.episode_index,
                    "seed": trace.seed,
                    "repeat_idx": trace.repeat_idx,
                    "episode_seed": trace.episode_seed,
                    "control_step": step,
                    "time_sec": f"{step / CONTROL_HZ:.6f}",
                    "terminal_step": trace.episode_steps,
                }
                row.update({name: f"{float(action[index]):.9g}" for index, name in enumerate(BODY_JOINTS_DFS[:15])})
                writer.writerow(row)
    return path


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    act_run = args.act_run.resolve()
    stream_run = args.stream_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    act_rows, act_median = select_near_median(read_summary(act_run), "success", args.count)
    stream_rows, stream_median = select_near_median(read_summary(stream_run), "fall", args.count)
    traces = [make_act_trace(act_run, row) for row in act_rows]
    traces.extend(make_stream_trace(stream_run, row) for row in stream_rows)

    generated = [plot_combined(traces, output_dir), write_trace_csv(traces, output_dir)]
    for trace in traces:
        generated.extend(plot_individual(trace, output_dir))

    manifest = {
        "selection_rule": "closest episode_steps to the outcome-class median; tie-break by episode_index",
        "control_hz": CONTROL_HZ,
        "joint_order": "DFS/MuJoCo",
        "act_run": str(act_run),
        "act_success_median_episode_steps": act_median,
        "stream_run": str(stream_run),
        "stream_fall_median_episode_steps": stream_median,
        "episodes": [
            {
                "source": trace.source,
                "outcome": trace.outcome,
                "episode_index": trace.episode_index,
                "seed": trace.seed,
                "repeat_idx": trace.repeat_idx,
                "episode_seed": trace.episode_seed,
                "terminal_step": trace.episode_steps,
                "trace_steps_including_post_termination": len(trace.actions),
                "source_trace": str(trace.source_trace),
                "video_path": str(trace.video_path),
            }
            for trace in traces
        ],
        "generated_files": [str(path.resolve()) for path in generated],
    }
    manifest_path = output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "episodes": manifest["episodes"]}, indent=2))


if __name__ == "__main__":
    main()
