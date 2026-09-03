#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


CHECKPOINTS = [70000, 80000, 100000, 120000, 160000]
HORIZONS = [1, 2, 3, 4, 5]
FLOWS = [5, 10, 20]
COLORS = {"off": "#3478bf", "on": "#e45756"}
LABELS = {"off": "Refiner off", "on": "Refiner on"}


def read_summary(path: Path):
    rows = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append({
                "checkpoint": int(row["checkpoint"]),
                "horizon": int(row["horizon"]),
                "flow": int(row["flow"]),
                "completed": int(row["completed"]),
                "successes": int(row["successes"]),
            })
    return rows


def wilson(successes, total, z=1.96):
    if not total:
        return 0.0, 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, max(0, center - half), min(1, center + half)


def setup_axis(ax, title):
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_series(ax, x, successes, totals, mode):
    stats = [wilson(s, n) for s, n in zip(successes, totals)]
    y = [v[0] for v in stats]
    lower = [v[0] - v[1] for v in stats]
    upper = [v[2] - v[0] for v in stats]
    ax.errorbar(
        x, y, yerr=[lower, upper], color=COLORS[mode], label=LABELS[mode],
        marker="o", markersize=5, linewidth=2, capsize=3,
    )


def aggregate(rows, **filters):
    selected = [row for row in rows if all(row[key] == value for key, value in filters.items())]
    return sum(row["successes"] for row in selected), sum(row["completed"] for row in selected)


def save_overall_horizon(data, output):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for mode in ("off", "on"):
        values = [aggregate(data[mode], horizon=h) for h in HORIZONS]
        plot_series(ax, HORIZONS, [v[0] for v in values], [v[1] for v in values], mode)
    setup_axis(ax, "OpenDoor success rate vs execution horizon\n(all checkpoints and Flow steps, 900 episodes per point)")
    ax.set_xlabel("Actions executed before replanning (horizon)")
    ax.set_ylabel("Success rate")
    ax.set_xticks(HORIZONS)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_horizon_facets(data, flow, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, checkpoint in zip(axes.flat, CHECKPOINTS):
        for mode in ("off", "on"):
            values = [aggregate(data[mode], checkpoint=checkpoint, horizon=h, flow=flow) for h in HORIZONS]
            plot_series(ax, HORIZONS, [v[0] for v in values], [v[1] for v in values], mode)
        setup_axis(ax, f"Checkpoint {checkpoint:06d}")
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Success rate")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=12)
    fig.suptitle(f"OpenDoor: horizon sweep with Flow steps={flow}\nCheckpoint and Flow fixed within each panel; 60 episodes per point", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_flow_facets(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, checkpoint in zip(axes.flat, CHECKPOINTS):
        for mode in ("off", "on"):
            values = [aggregate(data[mode], checkpoint=checkpoint, horizon=1, flow=f) for f in FLOWS]
            plot_series(ax, FLOWS, [v[0] for v in values], [v[1] for v in values], mode)
        setup_axis(ax, f"Checkpoint {checkpoint:06d}")
        ax.set_xticks(FLOWS)
        ax.set_xlabel("Flow Matching denoising steps")
        ax.set_ylabel("Success rate")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=12)
    fig.suptitle("OpenDoor: Flow-step sweep at horizon=1\nCheckpoint and horizon fixed within each panel; 60 episodes per point", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_overall_checkpoint(data, output):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for mode in ("off", "on"):
        values = [aggregate(data[mode], checkpoint=checkpoint) for checkpoint in CHECKPOINTS]
        plot_series(ax, CHECKPOINTS, [v[0] for v in values], [v[1] for v in values], mode)
    setup_axis(ax, "OpenDoor success rate vs checkpoint\n(all horizons and Flow steps, 900 episodes per point)")
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel("Success rate")
    ax.set_xticks(CHECKPOINTS, [f"{checkpoint:06d}" for checkpoint in CHECKPOINTS])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_checkpoint_facets(data, flow, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, horizon in zip(axes.flat, HORIZONS):
        for mode in ("off", "on"):
            values = [aggregate(data[mode], checkpoint=checkpoint, horizon=horizon, flow=flow) for checkpoint in CHECKPOINTS]
            plot_series(ax, CHECKPOINTS, [v[0] for v in values], [v[1] for v in values], mode)
        setup_axis(ax, f"Horizon {horizon}")
        ax.set_xticks(CHECKPOINTS, [f"{checkpoint // 1000}k" for checkpoint in CHECKPOINTS])
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Success rate")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=12)
    fig.suptitle(f"OpenDoor: checkpoint sweep with Flow steps={flow}\nHorizon and Flow fixed within each panel; 60 episodes per point", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: plot_opendoor_grid.py OFF_SUMMARY ON_SUMMARY OUTPUT_DIR")
    data = {"off": read_summary(Path(sys.argv[1])), "on": read_summary(Path(sys.argv[2]))}
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_overall_horizon(data, output_dir / "01_overall_horizon_on_off.png")
    for index, flow in enumerate(FLOWS, start=2):
        save_horizon_facets(data, flow, output_dir / f"0{index}_horizon_flow{flow}_on_off.png")
    save_flow_facets(data, output_dir / "05_flow_horizon1_on_off.png")
    save_overall_checkpoint(data, output_dir / "06_overall_checkpoint_on_off.png")
    for index, flow in enumerate(FLOWS, start=7):
        save_checkpoint_facets(data, flow, output_dir / f"0{index}_checkpoint_flow{flow}_on_off.png")
    print(f"wrote 9 plots to {output_dir}")


if __name__ == "__main__":
    main()
