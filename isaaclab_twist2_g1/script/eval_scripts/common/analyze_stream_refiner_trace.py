#!/usr/bin/env python3
"""Visualize Stream base/refiner/final actions recorded by HTTP evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


JOINT_NAMES = [
    "L hip pitch", "L hip roll", "L hip yaw", "L knee", "L ankle pitch",
    "L ankle roll", "R hip pitch", "R hip roll", "R hip yaw", "R knee",
    "R ankle pitch", "R ankle roll", "waist yaw", "waist roll", "waist pitch",
    "L shoulder pitch", "L shoulder roll", "L shoulder yaw", "L elbow",
    "L wrist roll", "L wrist pitch", "L wrist yaw", "R shoulder pitch",
    "R shoulder roll", "R shoulder yaw", "R elbow", "R wrist roll",
    "R wrist pitch", "R wrist yaw",
]

FIELDS = {
    "final": "action.applied_action",
    "base": "action.applied_action_before_refiner",
    "refined": "action.refined_action",
    "residual": "action.refiner_applied_delta",
    "normalized_base": "action.normalized_base_action_before_refiner",
    "normalized_refined": "action.normalized_refined_action",
    "normalized_residual": "action.normalized_refiner_applied_delta",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--residual-limit", type=float, default=0.5)
    return parser.parse_args()


def load_summary(run_dir: Path):
    with (run_dir / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["repeat_idx"] = int(row["repeat_idx"])
        row["episode_steps"] = int(row["episode_steps"])
        row["max_steps"] = int(row["max_steps"])
    return rows


def body_tilt_deg(states: np.ndarray) -> np.ndarray:
    gravity = states[:, 61:64]
    norm = np.linalg.norm(gravity, axis=1)
    upright = np.divide(-gravity[:, 2], norm, out=np.ones_like(norm), where=norm > 0)
    return np.degrees(np.arccos(np.clip(upright, -1.0, 1.0)))


def load_trace(row: dict):
    trace_path = Path(row["vla_trace_path"])
    records = []
    with trace_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") != "infer_single":
                continue
            missing = [field for field in FIELDS.values() if field not in record]
            if missing:
                raise ValueError(f"{trace_path}:{line_number} missing {missing}")
            records.append(record)
    terminal = row["episode_steps"]
    if len(records) < terminal:
        raise ValueError(f"{trace_path}: {len(records)} actions for terminal step {terminal}")
    records = records[:terminal]
    arrays = {
        name: np.asarray([record[field] for record in records], dtype=np.float64)
        for name, field in FIELDS.items()
    }
    for name, values in arrays.items():
        if values.shape != (terminal, 29):
            raise ValueError(f"{trace_path}: {name} has shape {values.shape}")
    states = np.asarray([record["observation_state"] for record in records], dtype=np.float64)
    if states.shape != (terminal, 93):
        raise ValueError(f"{trace_path}: observation state has shape {states.shape}")
    arrays.update(
        {
            "states": states,
            "tilt": body_tilt_deg(states),
            "angular_velocity": np.linalg.norm(states[:, 58:61], axis=1),
            "chunk_start": np.asarray([bool(record.get("chunk_start")) for record in records]),
            "chunk_index": np.asarray([int(record.get("chunk_index", -1)) for record in records]),
            "chunk_step_index": np.asarray([int(record.get("chunk_step_index", -1)) for record in records]),
        }
    )
    # These identities make the interpretation of the recorded fields explicit.
    if not np.allclose(arrays["final"], arrays["refined"], rtol=1e-6, atol=1e-7):
        raise ValueError(f"{trace_path}: final action does not match refined action")
    if not np.allclose(
        arrays["refined"] - arrays["base"], arrays["residual"], rtol=1e-6, atol=1e-7
    ):
        raise ValueError(f"{trace_path}: postprocessed residual identity failed")
    if not np.allclose(
        arrays["normalized_refined"] - arrays["normalized_base"],
        arrays["normalized_residual"],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(f"{trace_path}: normalized residual identity failed")
    return arrays


def safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        np.sum(left * right, axis=1), denom, out=np.full(len(left), np.nan), where=denom > 1e-12
    )


def first_offset(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0] - (len(mask) - 1)) if len(indices) else None


def episode_metrics(row: dict, data: dict, residual_limit: float):
    delta_base = np.diff(data["base"], axis=0)
    delta_final = np.diff(data["final"], axis=0)
    delta_residual = np.diff(data["residual"], axis=0)
    base_l2 = np.linalg.norm(delta_base, axis=1)
    final_l2 = np.linalg.norm(delta_final, axis=1)
    residual_l2 = np.linalg.norm(data["normalized_residual"], axis=1)
    normalized_base_abs = np.abs(data["normalized_base"])
    boundary = data["chunk_start"][1:]
    saturation = np.abs(data["normalized_residual"]) >= residual_limit * 0.98
    early_end = max(1, len(residual_l2) - 100)
    peak_index = int(np.argmax(final_l2)) if len(final_l2) else 0
    boundary_cosine = safe_cosine(delta_base[boundary], delta_residual[boundary])
    metric = {
        "seed": row["seed"],
        "repeat_idx": row["repeat_idx"],
        "failure_reason": row["failure_reason"],
        "episode_steps": row["episode_steps"],
        "base_delta_p95": float(np.percentile(base_l2, 95)),
        "final_delta_p95": float(np.percentile(final_l2, 95)),
        "boundary_base_delta_p95": float(np.percentile(base_l2[boundary], 95)),
        "boundary_final_delta_p95": float(np.percentile(final_l2[boundary], 95)),
        "boundary_smoothing_rate": float(np.mean(final_l2[boundary] < base_l2[boundary])),
        "within_chunk_smoothing_rate": float(np.mean(final_l2[~boundary] < base_l2[~boundary])),
        "boundary_delta_residual_vs_base_cosine_median": float(np.nanmedian(boundary_cosine)),
        "normalized_residual_l2_early_p50": float(np.median(residual_l2[:early_end])),
        "normalized_residual_l2_last100_p50": float(np.median(residual_l2[-100:])),
        "normalized_residual_component_saturation_rate": float(np.mean(saturation)),
        "normalized_residual_frame_saturation_rate": float(np.mean(np.any(saturation, axis=1))),
        "last100_residual_frame_saturation_rate": float(np.mean(np.any(saturation[-100:], axis=1))),
        "last100_lower_body_base_outside_unit_component_rate": float(
            np.mean(normalized_base_abs[-100:, :15] > 1.0)
        ),
        "last100_lower_body_base_abs_p99": float(np.percentile(normalized_base_abs[-100:, :15], 99)),
        "tilt10_first_offset": first_offset(data["tilt"] >= 10.0),
        "tilt15_first_offset": first_offset(data["tilt"] >= 15.0),
        "residual_saturation_first_offset": first_offset(np.any(saturation, axis=1)),
        "max_final_delta_offset": int(peak_index + 1 - (len(data["final"]) - 1)),
        "max_final_delta_at_chunk_boundary": bool(boundary[peak_index]),
        "max_tilt_deg": float(np.max(data["tilt"])),
        "trace_path": row["vla_trace_path"],
    }
    data.update(
        {
            "delta_base": delta_base,
            "delta_final": delta_final,
            "delta_residual": delta_residual,
            "base_delta_l2": base_l2,
            "final_delta_l2": final_l2,
            "normalized_residual_l2": residual_l2,
            "boundary_delta": boundary,
            "saturation": saturation,
        }
    )
    return metric


def load_run(run_dir: Path, residual_limit: float):
    episodes = []
    metrics = []
    for index, row in enumerate(load_summary(run_dir), start=1):
        data = load_trace(row)
        metric = episode_metrics(row, data, residual_limit)
        episodes.append({"row": row, "data": data, "metric": metric})
        metrics.append(metric)
        if index % 10 == 0:
            print(f"loaded {index} episodes", flush=True)
    return episodes, metrics


def aligned_matrix(episodes, key: str, window: int, *, delta: bool = False):
    matrix = np.full((len(episodes), window + 1), np.nan)
    for index, episode in enumerate(episodes):
        values = episode["data"][key]
        if values.ndim != 1:
            raise ValueError(f"aligned key {key} must be one-dimensional")
        # Frame series and transition series both end at the terminal frame;
        # the first delta is the transition into frame 1.
        offsets = np.arange(len(values)) - (len(values) - 1)
        keep = (offsets >= -window) & (offsets <= 0)
        matrix[index, offsets[keep] + window] = values[keep]
    return matrix


def aligned_joint_matrix(episodes, key: str, window: int):
    matrices = []
    for episode in episodes:
        values = episode["data"][key]
        matrix = np.full((window + 1, 29), np.nan)
        offsets = np.arange(len(values)) - (len(values) - 1)
        keep = (offsets >= -window) & (offsets <= 0)
        matrix[offsets[keep] + window] = values[keep]
        matrices.append(matrix)
    return np.stack(matrices)


def median_iqr(axis, x, matrix, label, color):
    median = np.nanmedian(matrix, axis=0)
    low = np.nanpercentile(matrix, 25, axis=0)
    high = np.nanpercentile(matrix, 75, axis=0)
    axis.plot(x, median, label=label, color=color)
    axis.fill_between(x, low, high, color=color, alpha=0.18)


def plot_aggregate(episodes, output_path: Path, window: int, residual_limit: float):
    falls = [episode for episode in episodes if episode["row"]["failure_reason"] == "fall"]
    x = np.arange(-window, 1)
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)

    tilt = aligned_matrix(falls, "tilt", window)
    angular = aligned_matrix(falls, "angular_velocity", window)
    median_iqr(axes[0, 0], x, tilt, "body tilt", "#c44e52")
    angular_axis = axes[0, 0].twinx()
    median_iqr(angular_axis, x, angular, "angular velocity", "#55a868")
    axes[0, 0].axhline(10, color="#c44e52", linestyle=":", linewidth=1)
    axes[0, 0].set_ylabel("tilt (deg)")
    angular_axis.set_ylabel("angular velocity L2")
    axes[0, 0].set_title("Robot state aligned to fall")

    base_delta = aligned_matrix(falls, "base_delta_l2", window, delta=True)
    final_delta = aligned_matrix(falls, "final_delta_l2", window, delta=True)
    median_iqr(axes[0, 1], x, base_delta, "base temporal delta", "#4c72b0")
    median_iqr(axes[0, 1], x, final_delta, "final temporal delta", "#dd8452")
    axes[0, 1].set_ylabel("action delta L2")
    axes[0, 1].set_title("Does the Refiner smooth base-action changes?")
    axes[0, 1].legend()

    residual = aligned_matrix(falls, "normalized_residual_l2", window)
    saturation_aligned = np.full((len(falls), window + 1), np.nan)
    for index, episode in enumerate(falls):
        values = np.any(episode["data"]["saturation"], axis=1).astype(float)
        offsets = np.arange(len(values)) - (len(values) - 1)
        keep = offsets >= -window
        saturation_aligned[index, offsets[keep] + window] = values[keep]
    median_iqr(axes[1, 0], x, residual, "normalized residual L2", "#8172b2")
    sat_axis = axes[1, 0].twinx()
    sat_axis.plot(x, np.nanmean(saturation_aligned, axis=0), color="#937860", label="any dim saturated")
    axes[1, 0].set_ylabel("normalized residual L2")
    sat_axis.set_ylabel("fraction of falls with saturation")
    sat_axis.set_ylim(0, 1.05)
    axes[1, 0].set_title(f"Refiner effort and saturation (limit={residual_limit:g})")

    all_boundary_base = np.concatenate([e["data"]["base_delta_l2"][e["data"]["boundary_delta"]] for e in falls])
    all_boundary_final = np.concatenate([e["data"]["final_delta_l2"][e["data"]["boundary_delta"]] for e in falls])
    all_inside_base = np.concatenate([e["data"]["base_delta_l2"][~e["data"]["boundary_delta"]] for e in falls])
    all_inside_final = np.concatenate([e["data"]["final_delta_l2"][~e["data"]["boundary_delta"]] for e in falls])
    axes[1, 1].boxplot(
        [all_boundary_base, all_boundary_final, all_inside_base, all_inside_final],
        labels=["boundary\nbase", "boundary\nfinal", "inside\nbase", "inside\nfinal"],
        showfliers=False,
    )
    axes[1, 1].set_ylabel("temporal action delta L2")
    axes[1, 1].set_title("Base vs final action discontinuity")

    residual_joint = aligned_joint_matrix(falls, "normalized_residual", window)
    median_joint = np.nanmedian(residual_joint, axis=0).T
    vmax = max(0.05, float(np.nanpercentile(np.abs(median_joint), 99)))
    image = axes[2, 0].imshow(
        median_joint, aspect="auto", origin="lower", interpolation="nearest",
        extent=[-window, 0, -0.5, 28.5], cmap="coolwarm", vmin=-vmax, vmax=vmax,
    )
    axes[2, 0].set_yticks(range(29))
    axes[2, 0].set_yticklabels(JOINT_NAMES, fontsize=7)
    axes[2, 0].set_xlabel("control steps relative to fall")
    axes[2, 0].set_title("Median normalized Refiner residual by joint")
    fig.colorbar(image, ax=axes[2, 0], pad=0.01)

    smoothing_boundary = []
    smoothing_inside = []
    cosine_boundary = []
    for episode in falls:
        data = episode["data"]
        boundary = data["boundary_delta"]
        smoothing_boundary.extend((data["final_delta_l2"][boundary] < data["base_delta_l2"][boundary]).tolist())
        smoothing_inside.extend((data["final_delta_l2"][~boundary] < data["base_delta_l2"][~boundary]).tolist())
        cosine_boundary.extend(safe_cosine(data["delta_base"][boundary], data["delta_residual"][boundary]))
    axes[2, 1].bar(
        ["boundary smooth", "inside smooth", "boundary amplify"],
        [np.mean(smoothing_boundary), np.mean(smoothing_inside), 1.0 - np.mean(smoothing_boundary)],
        color=["#55a868", "#4c72b0", "#c44e52"],
    )
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].set_ylabel("fraction of transitions")
    axes[2, 1].set_title(
        "Refiner temporal effect; boundary cosine median={:.3f}".format(np.nanmedian(cosine_boundary))
    )

    for axis in axes.flat:
        axis.axvline(0, color="red", linestyle="--", linewidth=1)
        if axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
            axis.set_xlabel("control steps relative to fall")
    fig.suptitle(f"Stream Refiner diagnosis across {len(falls)} falls", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_episode_pdf(episodes, output_path: Path, window: int, residual_limit: float):
    falls = [episode for episode in episodes if episode["row"]["failure_reason"] == "fall"]
    with PdfPages(output_path) as pdf:
        for episode in falls:
            row, data, metric = episode["row"], episode["data"], episode["metric"]
            terminal = len(data["final"])
            start = max(0, terminal - window - 1)
            frame_offsets = np.arange(start, terminal) - (terminal - 1)
            delta_offsets = np.arange(start + 1, terminal) - (terminal - 1)
            boundary = data["chunk_start"][start + 1 : terminal]
            fig, axes = plt.subplots(4, 1, figsize=(15, 12), constrained_layout=True,
                                     gridspec_kw={"height_ratios": [1, 1, 1.8, 1.8]})

            axes[0].plot(frame_offsets, data["tilt"][start:], label="tilt (deg)", color="#c44e52")
            axes[0].plot(frame_offsets, data["angular_velocity"][start:], label="angular velocity L2", color="#55a868")
            axes[0].axhline(10, color="#c44e52", linestyle=":")
            axes[0].legend(fontsize=8)
            axes[0].set_title(
                f"seed={row['seed']} repeat={row['repeat_idx']} fall_step={terminal} "
                f"boundary_smooth={metric['boundary_smoothing_rate']:.2f} "
                f"last100_sat={metric['last100_residual_frame_saturation_rate']:.2f}"
            )

            base_l2 = data["base_delta_l2"][start:]
            final_l2 = data["final_delta_l2"][start:]
            axes[1].plot(delta_offsets, base_l2, label="base temporal delta", color="#4c72b0")
            axes[1].plot(delta_offsets, final_l2, label="final temporal delta", color="#dd8452")
            axes[1].scatter(delta_offsets[boundary], final_l2[boundary], s=10, color="black", label="chunk start")
            axes[1].legend(fontsize=8)
            axes[1].set_ylabel("delta L2")

            residual = data["normalized_residual"][start:].T
            image = axes[2].imshow(
                residual, aspect="auto", origin="lower", interpolation="nearest",
                extent=[frame_offsets[0], 0, -0.5, 28.5], cmap="coolwarm",
                vmin=-residual_limit, vmax=residual_limit,
            )
            axes[2].set_yticks(range(29))
            axes[2].set_yticklabels(JOINT_NAMES, fontsize=7)
            axes[2].set_title("Normalized Refiner applied residual")
            fig.colorbar(image, ax=axes[2], pad=0.01)

            action_difference = (data["final"] - data["base"])[start:].T
            vmax = max(0.1, float(np.percentile(np.abs(action_difference), 99)))
            image = axes[3].imshow(
                action_difference, aspect="auto", origin="lower", interpolation="nearest",
                extent=[frame_offsets[0], 0, -0.5, 28.5], cmap="coolwarm", vmin=-vmax, vmax=vmax,
            )
            axes[3].set_yticks(range(29))
            axes[3].set_yticklabels(JOINT_NAMES, fontsize=7)
            axes[3].set_title("Postprocessed final action - base action")
            axes[3].set_xlabel("control steps relative to fall")
            fig.colorbar(image, ax=axes[3], pad=0.01)
            for axis in axes:
                axis.axvline(0, color="red", linestyle="--", linewidth=1)
            pdf.savefig(fig)
            plt.close(fig)


def plot_representative_action_fields(episodes, output_path: Path, window: int):
    """Plot every action dimension for the fall nearest the median duration."""
    falls = sorted(
        (episode for episode in episodes if episode["row"]["failure_reason"] == "fall"),
        key=lambda episode: episode["row"]["episode_steps"],
    )
    episode = falls[len(falls) // 2]
    row, data = episode["row"], episode["data"]
    terminal = len(data["final"])
    start = max(0, terminal - window)
    offsets = np.arange(start, terminal) - (terminal - 1)
    chunk_offsets = offsets[data["chunk_start"][start:]]
    fig, axes = plt.subplots(6, 5, figsize=(19, 18), sharex=True, constrained_layout=True)
    for joint_index, axis in enumerate(axes.flat):
        if joint_index >= 29:
            axis.axis("off")
            continue
        base = data["base"][start:, joint_index]
        final = data["final"][start:, joint_index]
        residual = data["residual"][start:, joint_index]
        axis.plot(offsets, base, color="#4c72b0", linewidth=1.0, label="base")
        axis.plot(offsets, final, color="#dd8452", linewidth=1.0, label="final/refined")
        axis.plot(offsets, residual, color="#55a868", linewidth=0.8, alpha=0.85, label="refiner delta")
        for chunk_offset in chunk_offsets:
            axis.axvline(chunk_offset, color="0.85", linewidth=0.5)
        axis.axvline(0, color="red", linestyle="--", linewidth=0.8)
        axis.set_title(f"d{joint_index}: {JOINT_NAMES[joint_index]}", fontsize=9)
        axis.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7, loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("steps relative to fall")
    fig.suptitle(
        "Representative action fields (median-duration fall): "
        f"seed={row['seed']} repeat={row['repeat_idx']} fall_step={terminal}",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def summarize(episodes, metrics, residual_limit: float):
    falls = [episode for episode in episodes if episode["row"]["failure_reason"] == "fall"]
    fall_metrics = [episode["metric"] for episode in falls]
    boundary_base = np.concatenate([e["data"]["base_delta_l2"][e["data"]["boundary_delta"]] for e in falls])
    boundary_final = np.concatenate([e["data"]["final_delta_l2"][e["data"]["boundary_delta"]] for e in falls])
    inside_base = np.concatenate([e["data"]["base_delta_l2"][~e["data"]["boundary_delta"]] for e in falls])
    inside_final = np.concatenate([e["data"]["final_delta_l2"][~e["data"]["boundary_delta"]] for e in falls])
    residual_frames = np.concatenate([e["data"]["normalized_residual"] for e in falls])
    saturation = np.abs(residual_frames) >= residual_limit * 0.98
    boundary_cosines = np.concatenate([
        safe_cosine(e["data"]["delta_base"][e["data"]["boundary_delta"]],
                    e["data"]["delta_residual"][e["data"]["boundary_delta"]])
        for e in falls
    ])
    tilt10_offsets = [m["tilt10_first_offset"] for m in fall_metrics if m["tilt10_first_offset"] is not None]
    saturation_offsets = [
        m["residual_saturation_first_offset"] for m in fall_metrics
        if m["residual_saturation_first_offset"] is not None
    ]
    early_residual = [m["normalized_residual_l2_early_p50"] for m in fall_metrics]
    late_residual = [m["normalized_residual_l2_last100_p50"] for m in fall_metrics]
    peak_boundary = [m["max_final_delta_at_chunk_boundary"] for m in fall_metrics]
    outcome_comparison = {}
    for reason in sorted({metric["failure_reason"] for metric in metrics}):
        selected = [metric for metric in metrics if metric["failure_reason"] == reason]
        outcome_comparison[reason] = {
            "episodes": len(selected),
            "boundary_smoothing_rate_median": float(
                np.median([metric["boundary_smoothing_rate"] for metric in selected])
            ),
            "last100_residual_l2_p50_median": float(
                np.median([metric["normalized_residual_l2_last100_p50"] for metric in selected])
            ),
            "last100_residual_frame_saturation_rate_median": float(
                np.median([metric["last100_residual_frame_saturation_rate"] for metric in selected])
            ),
            "last100_lower_body_base_outside_unit_component_rate_median": float(
                np.median([
                    metric["last100_lower_body_base_outside_unit_component_rate"]
                    for metric in selected
                ])
            ),
            "last100_lower_body_base_abs_p99_median": float(
                np.median([metric["last100_lower_body_base_abs_p99"] for metric in selected])
            ),
        }
    return {
        "episodes": len(episodes),
        "outcomes": dict(Counter(e["row"]["failure_reason"] for e in episodes)),
        "falls_analyzed": len(falls),
        "residual_limit": residual_limit,
        "boundary": {
            "base_delta_p50": float(np.median(boundary_base)),
            "base_delta_p95": float(np.percentile(boundary_base, 95)),
            "final_delta_p50": float(np.median(boundary_final)),
            "final_delta_p95": float(np.percentile(boundary_final, 95)),
            "final_to_base_delta_p50_ratio": float(np.median(boundary_final) / np.median(boundary_base)),
            "smoothing_transition_rate": float(np.mean(boundary_final < boundary_base)),
            "delta_residual_vs_delta_base_cosine_median": float(np.nanmedian(boundary_cosines)),
            "fall_peak_at_chunk_boundary_rate": float(np.mean(peak_boundary)),
        },
        "within_chunk": {
            "base_delta_p50": float(np.median(inside_base)),
            "base_delta_p95": float(np.percentile(inside_base, 95)),
            "final_delta_p50": float(np.median(inside_final)),
            "final_delta_p95": float(np.percentile(inside_final, 95)),
            "final_to_base_delta_p50_ratio": float(np.median(inside_final) / np.median(inside_base)),
            "smoothing_transition_rate": float(np.mean(inside_final < inside_base)),
        },
        "refiner_effort": {
            "component_saturation_rate": float(np.mean(saturation)),
            "frame_any_component_saturation_rate": float(np.mean(np.any(saturation, axis=1))),
            "falls_last100_frame_saturation_rate_median": float(
                np.median([m["last100_residual_frame_saturation_rate"] for m in fall_metrics])
            ),
            "early_residual_l2_p50_median": float(np.median(early_residual)),
            "last100_residual_l2_p50_median": float(np.median(late_residual)),
            "last100_to_early_residual_ratio": float(np.median(late_residual) / np.median(early_residual)),
            "first_saturation_offset_from_fall_median": float(np.median(saturation_offsets)),
        },
        "state_timing": {
            "first_tilt10_offset_from_fall_median": float(np.median(tilt10_offsets)),
            "saturation_minus_tilt10_median_steps": float(
                np.median([
                    m["residual_saturation_first_offset"] - m["tilt10_first_offset"]
                    for m in fall_metrics
                    if m["residual_saturation_first_offset"] is not None and m["tilt10_first_offset"] is not None
                ])
            ),
        },
        "outcome_comparison": outcome_comparison,
        "interpretation_note": (
            "Base actions are counterfactual one-step commands on states generated under the Refiner. "
            "This decomposition measures immediate smoothing/amplification, not a no-Refiner rollout outcome."
        ),
    }


def write_metrics(path: Path, metrics: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes, metrics = load_run(args.run_dir, args.residual_limit)
    write_metrics(args.output_dir / "episode_refiner_metrics.csv", metrics)
    report = summarize(episodes, metrics, args.residual_limit)
    with (args.output_dir / "refiner_analysis_summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    plot_aggregate(episodes, args.output_dir / "aggregate_refiner_diagnosis.png", args.window, args.residual_limit)
    plot_representative_action_fields(
        episodes, args.output_dir / "representative_fall_action_fields.png", args.window
    )
    plot_episode_pdf(episodes, args.output_dir / "all_falls_refiner_diagnosis.pdf", args.window, args.residual_limit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
