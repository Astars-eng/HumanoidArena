#!/usr/bin/env python3
"""Compare two stream evaluations using the actions recorded by the VLA server.

The server log contains the final action sent to the simulator at every control
step.  This tool reconstructs episodes, computes temporal action deltas, and
aligns them to the terminal frame.  It cannot reconstruct the refiner's
internal residual because base and refined actions are not logged separately.
"""

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ACTION_RE = re.compile(r"\[lerobot_vla_server\] infer count=(\d+) action=(\[.*\])$")
STATE_RE = re.compile(
    r"\[lerobot_vla_server\]\[state_debug\] infer=(\d+) raw_state shape=.* first=(\[.*\])$"
)
TASK_RE = re.compile(r"task_(\d+)__")

JOINT_NAMES = [
    "L hip pitch", "L hip roll", "L hip yaw", "L knee", "L ankle pitch",
    "L ankle roll", "R hip pitch", "R hip roll", "R hip yaw", "R knee",
    "R ankle pitch", "R ankle roll", "waist yaw", "waist roll", "waist pitch",
    "L shoulder pitch", "L shoulder roll", "L shoulder yaw", "L elbow",
    "L wrist roll", "L wrist pitch", "L wrist yaw", "R shoulder pitch",
    "R shoulder roll", "R shoulder yaw", "R elbow", "R wrist roll",
    "R wrist pitch", "R wrist yaw",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--window", type=int, default=200)
    return parser.parse_args()


def load_summary(run_dir):
    with (run_dir / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["repeat_idx"] = int(row["repeat_idx"])
        row["episode_steps"] = int(row["episode_steps"])
        row["max_steps"] = int(row["max_steps"])
    return rows


def parse_server_log(path):
    episodes = []
    current = None
    with path.open(errors="replace") as handle:
        for line in handle:
            if "[lerobot_vla_server] first_infer " in line:
                if current is not None and current["actions"]:
                    episodes.append(current)
                current = {"actions": [], "states": {}}
                continue
            state_match = STATE_RE.search(line.rstrip())
            if state_match:
                if current is None:
                    current = {"actions": [], "states": {}}
                values = ast.literal_eval(state_match.group(2))
                if len(values) >= 64:
                    current["states"][int(state_match.group(1))] = values
                continue
            action_match = ACTION_RE.search(line.rstrip())
            if not action_match:
                continue
            count = int(action_match.group(1))
            if current is None or (count == 1 and current["actions"]):
                if current is not None and current["actions"]:
                    episodes.append(current)
                current = {"actions": [], "states": {}}
            action = ast.literal_eval(action_match.group(2))
            if len(action) != 29:
                raise ValueError("Expected 29 actions in {} count {}, got {}".format(path, count, len(action)))
            current["actions"].append(action)
    if current is not None and current["actions"]:
        episodes.append(current)
    return episodes


def load_run(run_dir, label, chunk_size):
    rows = load_summary(run_dir)
    rows_by_seed = defaultdict(list)
    for row in rows:
        rows_by_seed[row["seed"]].append(row)
    records = []
    diagnostics = []
    log_paths = sorted((run_dir / "logs").glob("server__*.log"))
    if not log_paths:
        raise FileNotFoundError("No server logs under {}".format(run_dir / "logs"))
    for log_path in log_paths:
        match = TASK_RE.search(log_path.name)
        if not match:
            continue
        seed = int(match.group(1))
        seed_rows = sorted(rows_by_seed.get(seed, []), key=lambda row: row["repeat_idx"])
        segments = parse_server_log(log_path)
        diagnostics.append({
            "seed": seed,
            "summary_episodes": len(seed_rows),
            "logged_segments": len(segments),
            "logged_action_counts": [len(segment["actions"]) for segment in segments],
        })
        if len(segments) < len(seed_rows):
            raise ValueError(
                "{} seed {}: {} summary rows but only {} action segments".format(
                    label, seed, len(seed_rows), len(segments)
                )
            )
        for row, segment in zip(seed_rows, segments):
            terminal = row["episode_steps"]
            if len(segment["actions"]) < terminal:
                raise ValueError(
                    "{} seed {} repeat {}: terminal step {}, only {} logged actions".format(
                        label, seed, row["repeat_idx"], terminal, len(segment["actions"])
                    )
                )
            actions = np.asarray(segment["actions"][:terminal], dtype=np.float64)
            delta = np.diff(actions, axis=0)
            delta_l2 = np.linalg.norm(delta, axis=1)
            delta_max_abs = np.max(np.abs(delta), axis=1)
            delta_steps = np.arange(2, terminal + 1)
            boundary = ((delta_steps - 1) % chunk_size) == 0
            states = {}
            for step, values in segment["states"].items():
                if step <= terminal:
                    gravity = np.asarray(values[61:64], dtype=np.float64)
                    gravity_norm = np.linalg.norm(gravity)
                    if gravity_norm > 0:
                        upright_cos = np.clip(-gravity[2] / gravity_norm, -1.0, 1.0)
                        tilt_deg = math.degrees(math.acos(upright_cos))
                    else:
                        tilt_deg = float("nan")
                    states[step] = {
                        "tilt_deg": tilt_deg,
                        "angular_velocity_l2": float(np.linalg.norm(values[58:61])),
                    }
            records.append({
                "label": label,
                "run_dir": str(run_dir.resolve()),
                "row": row,
                "actions": actions,
                "delta": delta,
                "delta_l2": delta_l2,
                "delta_max_abs": delta_max_abs,
                "delta_steps": delta_steps,
                "boundary": boundary,
                "states": states,
            })
    found_seeds = {record["row"]["seed"] for record in records}
    missing = sorted(set(rows_by_seed) - found_seeds)
    if missing:
        raise ValueError("No server log matched summary seeds {} for {}".format(missing, label))
    return records, diagnostics


def percentile_rank(reference, value):
    if len(reference) == 0 or not np.isfinite(value):
        return float("nan")
    return 100.0 * float(np.count_nonzero(reference <= value)) / len(reference)


def build_metrics(records):
    all_delta = np.concatenate([r["delta_l2"] for r in records if len(r["delta_l2"])])
    threshold99 = float(np.percentile(all_delta, 99))
    metrics = []
    for record in records:
        row = record["row"]
        values = record["delta_l2"]
        max_index = int(np.argmax(values)) if len(values) else -1
        last20 = values[-20:]
        last50 = values[-50:]
        early = values[:-50] if len(values) > 50 else values
        delta_abs = np.abs(record["delta"])
        if delta_abs.size:
            joint_flat = int(np.argmax(delta_abs))
            joint_index = joint_flat % delta_abs.shape[1]
            peak_delta = record["delta"][max_index]
            peak_lower_energy_fraction = float(
                np.sum(np.square(peak_delta[:15])) / max(np.sum(np.square(peak_delta)), 1e-12)
            )
        else:
            joint_index = -1
            peak_lower_energy_fraction = float("nan")
        state_steps = sorted(record["states"])
        last_state = record["states"][state_steps[-1]] if state_steps else {}
        metric = {
            "label": record["label"],
            "seed": row["seed"],
            "repeat_idx": row["repeat_idx"],
            "failure_reason": row["failure_reason"],
            "success": row["success"],
            "episode_steps": row["episode_steps"],
            "max_steps": row["max_steps"],
            "delta_l2_p50": float(np.percentile(values, 50)) if len(values) else float("nan"),
            "delta_l2_p95": float(np.percentile(values, 95)) if len(values) else float("nan"),
            "delta_l2_max": float(values[max_index]) if max_index >= 0 else float("nan"),
            "max_delta_offset_from_terminal": int(record["delta_steps"][max_index] - row["episode_steps"]) if max_index >= 0 else 0,
            "max_delta_at_chunk_boundary": bool(record["boundary"][max_index]) if max_index >= 0 else False,
            "max_delta_lower_body_energy_fraction": peak_lower_energy_fraction,
            "terminal_delta_l2": float(values[-1]) if len(values) else float("nan"),
            "terminal_delta_percentile": percentile_rank(early, values[-1]) if len(values) else float("nan"),
            "last20_delta_l2_max": float(np.max(last20)) if len(last20) else float("nan"),
            "last20_has_runwide_p99_jump": bool(np.any(last20 >= threshold99)),
            "last50_to_early_p95_ratio": (
                float(np.percentile(last50, 95) / max(np.percentile(early, 95), 1e-12))
                if len(last50) and len(early) else float("nan")
            ),
            "largest_abs_delta_joint": JOINT_NAMES[joint_index] if joint_index >= 0 else "",
            "boundary_delta_p95": float(np.percentile(values[record["boundary"]], 95)) if np.any(record["boundary"]) else float("nan"),
            "inside_chunk_delta_p95": float(np.percentile(values[~record["boundary"]], 95)) if np.any(~record["boundary"]) else float("nan"),
            "last_logged_state_step": state_steps[-1] if state_steps else "",
            "last_logged_tilt_deg": last_state.get("tilt_deg", float("nan")),
            "last_logged_angular_velocity_l2": last_state.get("angular_velocity_l2", float("nan")),
        }
        metrics.append(metric)
    return metrics, threshold99


def aligned_delta(records, window):
    selected = [r for r in records if r["row"]["failure_reason"] == "fall"]
    matrix = np.full((len(selected), window + 1), np.nan)
    for index, record in enumerate(selected):
        offsets = record["delta_steps"] - record["row"]["episode_steps"]
        keep = (offsets >= -window) & (offsets <= 0)
        matrix[index, offsets[keep] + window] = record["delta_l2"][keep]
    return matrix


def aggregate_stats(records, metrics, threshold99):
    reasons = Counter(r["row"]["failure_reason"] for r in records)
    falls = [m for m in metrics if m["failure_reason"] == "fall"]
    boundary = np.concatenate([r["delta_l2"][r["boundary"]] for r in records])
    inside = np.concatenate([r["delta_l2"][~r["boundary"]] for r in records])
    return {
        "episodes": len(records),
        "reason_counts": dict(reasons),
        "fall_rate": reasons.get("fall", 0) / len(records),
        "max_steps_values": sorted(set(r["row"]["max_steps"] for r in records)),
        "fall_terminal_step_mean": float(np.mean([m["episode_steps"] for m in falls])),
        "fall_terminal_step_median": float(np.median([m["episode_steps"] for m in falls])),
        "runwide_delta_l2_p99": threshold99,
        "falls_with_p99_jump_in_last20": int(sum(m["last20_has_runwide_p99_jump"] for m in falls)),
        "falls_with_p99_jump_in_last20_rate": float(np.mean([m["last20_has_runwide_p99_jump"] for m in falls])),
        "terminal_delta_percentile_median": float(np.median([m["terminal_delta_percentile"] for m in falls])),
        "last50_to_early_p95_ratio_median": float(np.median([m["last50_to_early_p95_ratio"] for m in falls])),
        "falls_max_delta_at_chunk_boundary_rate": float(np.mean([m["max_delta_at_chunk_boundary"] for m in falls])),
        "fall_peak_delta_lower_body_energy_fraction_median": float(
            np.median([m["max_delta_lower_body_energy_fraction"] for m in falls])
        ),
        "chunk_boundary_delta_p50": float(np.percentile(boundary, 50)),
        "chunk_boundary_delta_p95": float(np.percentile(boundary, 95)),
        "inside_chunk_delta_p50": float(np.percentile(inside, 50)),
        "inside_chunk_delta_p95": float(np.percentile(inside, 95)),
    }


def causal_timing_stats(records, threshold99):
    """Compare first large action jump with first logged body-tilt crossing."""
    falls = [r for r in records if r["row"]["failure_reason"] == "fall"]
    result = {}
    for tilt_threshold in (10, 15, 20, 30):
        counts = Counter()
        jump_minus_tilt = []
        tilt_offsets = []
        jump_offsets = []
        for record in falls:
            terminal = record["row"]["episode_steps"]
            tilt_steps = [
                step for step in sorted(record["states"])
                if record["states"][step]["tilt_deg"] >= tilt_threshold
            ]
            jump_indices = np.flatnonzero(record["delta_l2"] >= threshold99)
            if not tilt_steps or not len(jump_indices):
                counts["missing"] += 1
                continue
            tilt_step = tilt_steps[0]
            jump_step = int(record["delta_steps"][jump_indices[0]])
            if jump_step < tilt_step:
                counts["jump_before_tilt"] += 1
            elif jump_step > tilt_step:
                counts["jump_after_tilt"] += 1
            else:
                counts["same_logged_step"] += 1
            jump_minus_tilt.append(jump_step - tilt_step)
            tilt_offsets.append(tilt_step - terminal)
            jump_offsets.append(jump_step - terminal)
        result[str(tilt_threshold)] = {
            "counts": dict(counts),
            "median_jump_step_minus_tilt_step": float(np.median(jump_minus_tilt)),
            "median_tilt_onset_offset_from_fall": float(np.median(tilt_offsets)),
            "median_p99_jump_offset_from_fall": float(np.median(jump_offsets)),
        }
    return result


def plot_aggregate(groups, metrics_by_label, output_path, window):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    labels = list(groups)
    colors = ["#4776b4", "#e1812c"]

    reason_names = sorted(set().union(*[Counter(r["row"]["failure_reason"] for r in groups[label]) for label in labels]))
    bottom = np.zeros(len(labels))
    for reason in reason_names:
        values = np.asarray([
            sum(r["row"]["failure_reason"] == reason for r in groups[label]) / len(groups[label])
            for label in labels
        ])
        axes[0, 0].bar(labels, values, bottom=bottom, label=reason)
        bottom += values
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("episode fraction")
    axes[0, 0].set_title("Evaluation outcomes")
    axes[0, 0].legend()

    fall_steps = [
        [r["row"]["episode_steps"] for r in groups[label] if r["row"]["failure_reason"] == "fall"]
        for label in labels
    ]
    axes[0, 1].boxplot(fall_steps, labels=labels, showmeans=True)
    axes[0, 1].set_ylabel("terminal control step")
    axes[0, 1].set_title("Fall time (50 Hz control)")

    x = np.arange(-window, 1)
    for label, color in zip(labels, colors):
        matrix = aligned_delta(groups[label], window)
        median = np.nanmedian(matrix, axis=0)
        low = np.nanpercentile(matrix, 25, axis=0)
        high = np.nanpercentile(matrix, 75, axis=0)
        axes[1, 0].plot(x, median, label=label, color=color)
        axes[1, 0].fill_between(x, low, high, color=color, alpha=0.2)
    axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=1, label="fall")
    axes[1, 0].set_xlabel("control steps relative to fall")
    axes[1, 0].set_ylabel("||action[t] - action[t-1]||2")
    axes[1, 0].set_title("Action delta aligned to fall (median and IQR)")
    axes[1, 0].legend()

    positions = []
    values = []
    tick_labels = []
    for group_index, label in enumerate(labels):
        boundary = np.concatenate([r["delta_l2"][r["boundary"]] for r in groups[label]])
        inside = np.concatenate([r["delta_l2"][~r["boundary"]] for r in groups[label]])
        positions.extend([group_index * 3 + 1, group_index * 3 + 2])
        values.extend([boundary, inside])
        tick_labels.extend([label + "\nboundary", label + "\ninside"])
    axes[1, 1].boxplot(values, positions=positions, labels=tick_labels, showfliers=False)
    axes[1, 1].set_ylabel("||action[t] - action[t-1]||2")
    axes[1, 1].set_title("5-step chunk boundary vs within-chunk delta")

    fig.suptitle("Stream action-delta comparison (logged final emitted actions)", fontsize=15)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_episode_pdf(records, metrics, output_path, window, chunk_size):
    metrics_by_key = {(m["seed"], m["repeat_idx"]): m for m in metrics}
    falls = [r for r in records if r["row"]["failure_reason"] == "fall"]
    with PdfPages(output_path) as pdf:
        for record in falls:
            row = record["row"]
            terminal = row["episode_steps"]
            start = max(0, terminal - window - 1)
            actions = record["actions"][start:terminal]
            delta = np.diff(actions, axis=0)
            delta_l2 = np.linalg.norm(delta, axis=1)
            steps = np.arange(start + 2, terminal + 1)
            offsets = steps - terminal
            metric = metrics_by_key[(row["seed"], row["repeat_idx"])]

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True,
                                     gridspec_kw={"height_ratios": [1.2, 2.2, 1.0]})
            axes[0].plot(offsets, delta_l2, color="#4776b4", linewidth=1.2)
            boundary = ((steps - 1) % chunk_size) == 0
            axes[0].scatter(offsets[boundary], delta_l2[boundary], s=9, color="#e1812c",
                            label="new 5-step execution chunk")
            axes[0].axvline(0, color="red", linestyle="--", label="fall")
            axes[0].set_ylabel("delta L2")
            axes[0].legend(loc="upper left", fontsize=8)
            axes[0].set_title(
                "{} seed={} repeat={} fall_step={} | peak_offset={} | last20_p99_jump={}".format(
                    record["label"], row["seed"], row["repeat_idx"], terminal,
                    metric["max_delta_offset_from_terminal"],
                    metric["last20_has_runwide_p99_jump"],
                )
            )

            image = axes[1].imshow(
                delta.T, aspect="auto", interpolation="nearest", origin="lower",
                extent=[offsets[0], offsets[-1], -0.5, 28.5], cmap="coolwarm",
                vmin=-max(0.3, float(np.percentile(np.abs(delta), 99))),
                vmax=max(0.3, float(np.percentile(np.abs(delta), 99))),
            )
            axes[1].set_yticks(range(29))
            axes[1].set_yticklabels(JOINT_NAMES, fontsize=7)
            axes[1].set_ylabel("action dimension")
            axes[1].set_title("Per-joint temporal action delta")
            fig.colorbar(image, ax=axes[1], pad=0.01, label="action[t] - action[t-1]")

            state_steps = np.asarray(sorted(record["states"]), dtype=int)
            if len(state_steps):
                state_offsets = state_steps - terminal
                keep = state_offsets >= -window
                state_offsets = state_offsets[keep]
                tilt = [record["states"][int(s)]["tilt_deg"] for s in state_steps[keep]]
                angular = [record["states"][int(s)]["angular_velocity_l2"] for s in state_steps[keep]]
                axes[2].plot(state_offsets, tilt, marker="o", markersize=3, label="body tilt (deg)")
                angular_axis = axes[2].twinx()
                angular_axis.plot(state_offsets, angular, marker=".", color="#55a868",
                                  label="angular velocity L2")
                angular_axis.set_ylabel("angular velocity L2", color="#55a868")
            axes[2].axvline(0, color="red", linestyle="--")
            axes[2].set_xlabel("control steps relative to fall")
            axes[2].set_ylabel("body tilt (deg)")
            axes[2].set_title("Sparse proprioception recovered from server state_debug")
            pdf.savefig(fig)
            plt.close(fig)


def paired_stats(groups):
    labels = list(groups)
    left = {(r["row"]["seed"], r["row"]["repeat_idx"]): r for r in groups[labels[0]]}
    right = {(r["row"]["seed"], r["row"]["repeat_idx"]): r for r in groups[labels[1]]}
    keys = sorted(set(left) & set(right))
    both_fall = [key for key in keys if left[key]["row"]["failure_reason"] == "fall" and right[key]["row"]["failure_reason"] == "fall"]
    differences = [right[key]["row"]["episode_steps"] - left[key]["row"]["episode_steps"] for key in both_fall]
    transitions = Counter(
        left[key]["row"]["failure_reason"] + " -> " + right[key]["row"]["failure_reason"] for key in keys
    )
    return {
        "paired_episodes": len(keys),
        "outcome_transitions": dict(transitions),
        "both_fall_count": len(both_fall),
        "right_minus_left_fall_step_mean": float(np.mean(differences)) if differences else float("nan"),
        "right_minus_left_fall_step_median": float(np.median(differences)) if differences else float("nan"),
        "right_fell_later_count": int(sum(value > 0 for value in differences)),
        "same_fall_step_count": int(sum(value == 0 for value in differences)),
        "right_fell_earlier_count": int(sum(value < 0 for value in differences)),
    }


def write_metrics(path, metrics):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_a, diagnostics_a = load_run(args.run_a, args.label_a, args.chunk_size)
    records_b, diagnostics_b = load_run(args.run_b, args.label_b, args.chunk_size)
    groups = {args.label_a: records_a, args.label_b: records_b}
    metrics_a, threshold_a = build_metrics(records_a)
    metrics_b, threshold_b = build_metrics(records_b)
    metrics_by_label = {args.label_a: metrics_a, args.label_b: metrics_b}
    all_metrics = metrics_a + metrics_b

    write_metrics(args.output_dir / "episode_action_delta_metrics.csv", all_metrics)
    plot_aggregate(groups, metrics_by_label, args.output_dir / "aggregate_comparison.png", args.window)
    plot_episode_pdf(records_a, metrics_a, args.output_dir / "{}_falls.pdf".format(args.label_a), args.window, args.chunk_size)
    plot_episode_pdf(records_b, metrics_b, args.output_dir / "{}_falls.pdf".format(args.label_b), args.window, args.chunk_size)

    report = {
        "scope_note": (
            "Temporal action delta is computed from final emitted actions in the server logs. "
            "It is not the ActionDeltaRefiner internal residual, which was not logged."
        ),
        "chunk_size": args.chunk_size,
        "window_before_fall": args.window,
        "runs": {
            args.label_a: {
                **aggregate_stats(records_a, metrics_a, threshold_a),
                "causal_timing": causal_timing_stats(records_a, threshold_a),
            },
            args.label_b: {
                **aggregate_stats(records_b, metrics_b, threshold_b),
                "causal_timing": causal_timing_stats(records_b, threshold_b),
            },
        },
        "paired": paired_stats(groups),
        "log_alignment_diagnostics": {
            args.label_a: diagnostics_a,
            args.label_b: diagnostics_b,
        },
    }
    with (args.output_dir / "analysis_summary.json").open("w") as handle:
        json.dump(report, handle, indent=2, allow_nan=True)
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
