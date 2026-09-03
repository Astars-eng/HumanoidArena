#!/usr/bin/env python3
"""Plot initial football-position distributions from SONIC episode NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POSITION_KEYS = (
    "episode_init_env_obj_football_position",
    "env_obj_football_position",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def read_initial_position(path: Path) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        for key in POSITION_KEYS:
            if key not in data.files:
                continue
            value = np.asarray(data[key], dtype=np.float64)
            if value.ndim > 1:
                value = value[0]
            value = value.reshape(-1)
            if value.size >= 3 and np.all(np.isfinite(value[:3])):
                return value[:3], key
    raise KeyError(f"no valid football initial position in {path}")


def padded_limits(values: np.ndarray, fraction: float = 0.08) -> tuple[float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    pad = max(span * fraction, 0.02)
    return lo - pad, hi + pad


def main() -> None:
    args = parse_args()
    files = sorted(args.source_root.rglob("*.npz"))[: args.limit]
    if len(files) != args.limit:
        raise RuntimeError(f"expected {args.limit} NPZ files, found {len(files)}")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, path in enumerate(files, start=1):
        try:
            position, key = read_initial_position(path)
            rows.append(
                {
                    "index": index,
                    "source_group": path.parent.name,
                    "episode": path.stem,
                    "x_m": float(position[0]),
                    "y_m": float(position[1]),
                    "z_m": float(position[2]),
                    "position_key": key,
                    "source_path": str(path),
                }
            )
        except Exception as exc:
            failures.append({"source_path": str(path), "error": str(exc)})
        if index % 10 == 0:
            print(f"processed {index}/{len(files)}", flush=True)

    if len(rows) != args.limit:
        raise RuntimeError(
            f"only {len(rows)}/{args.limit} episodes had valid positions; failures={failures}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "football_initial_positions_100.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    xyz = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in rows])
    groups = sorted({str(row["source_group"]) for row in rows})
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for group_index, group in enumerate(groups):
        mask = np.asarray([row["source_group"] == group for row in rows])
        ax.scatter(
            xyz[mask, 0],
            xyz[mask, 1],
            s=42,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.5,
            color=colors(group_index),
            label=f"{group} (n={int(mask.sum())})",
        )
    ax.scatter(
        [xyz[:, 0].mean()],
        [xyz[:, 1].mean()],
        marker="X",
        s=180,
        color="black",
        label="mean",
        zorder=5,
    )
    ax.set(
        title="SONIC Football Initial Positions (100 Episodes)",
        xlabel="Initial X (m)",
        ylabel="Initial Y (m)",
        xlim=padded_limits(xyz[:, 0]),
        ylim=padded_limits(xyz[:, 1]),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    scatter_path = args.output_dir / "football_initial_xy_scatter.png"
    fig.savefig(scatter_path, dpi=220)
    fig.savefig(args.output_dir / "football_initial_xy_scatter.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, values, label, color in zip(
        axes.flat[:3],
        xyz.T,
        ("Initial X (m)", "Initial Y (m)", "Initial Z (m)"),
        ("#4c78a8", "#f58518", "#54a24b"),
    ):
        axis.hist(values, bins=12, color=color, alpha=0.82, edgecolor="white")
        axis.axvline(values.mean(), color="black", linestyle="--", linewidth=1.5, label="mean")
        axis.axvline(np.median(values), color="#b22222", linestyle=":", linewidth=1.5, label="median")
        axis.set(xlabel=label, ylabel="Episode count")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()

    hist2d = axes[1, 1].hist2d(xyz[:, 0], xyz[:, 1], bins=12, cmap="viridis")
    axes[1, 1].set(xlabel="Initial X (m)", ylabel="Initial Y (m)", title="2D count density")
    fig.colorbar(hist2d[3], ax=axes[1, 1], label="Episode count")
    fig.suptitle("SONIC Football Initial-Position Distribution (n=100)", fontsize=15)
    distribution_path = args.output_dir / "football_initial_position_distribution.png"
    fig.savefig(distribution_path, dpi=220)
    fig.savefig(args.output_dir / "football_initial_position_distribution.svg")
    plt.close(fig)

    axis_names = ("x_m", "y_m", "z_m")
    summary = {
        "source_root": str(args.source_root),
        "episode_count": len(rows),
        "source_group_counts": {
            group: sum(row["source_group"] == group for row in rows) for group in groups
        },
        "position_key_counts": {
            key: sum(row["position_key"] == key for row in rows)
            for key in sorted({str(row["position_key"]) for row in rows})
        },
        "statistics": {
            name: {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=0)),
            }
            for name, values in zip(axis_names, xyz.T)
        },
        "failures": failures,
    }
    summary_path = args.output_dir / "football_initial_position_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"scatter={scatter_path}")
    print(f"distribution={distribution_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
