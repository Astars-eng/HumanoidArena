from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.merge_humanoidarena_datasets import (
    DERIVED_VECTOR_FEATURES,
    STAT_NAMES,
    build_derived_feature_specs,
    concatenate_stats,
    concatenate_vector_columns,
    exact_feature_stats,
    repair_merged_stats,
    recompute_tabular_stats,
)


def _feature(dim: int, prefix: str) -> dict:
    return {
        "dtype": "float32",
        "shape": [dim],
        "names": [f"{prefix}_{index}" for index in range(dim)],
    }


def test_act_derived_features_include_last_action_and_raw107_action() -> None:
    features = {
        "observation.joint_pos": _feature(29, "joint_pos"),
        "observation.joint_vel": _feature(29, "joint_vel"),
        "observation.ang_vel_b": _feature(3, "ang_vel"),
        "observation.gravity": _feature(3, "gravity"),
        "observation.last_action": _feature(29, "last_action"),
        "action.applied_action": _feature(29, "applied_action"),
        "action.motion_token": _feature(64, "motion_token"),
        "action.left_hand": _feature(7, "left_hand"),
        "action.right_hand": _feature(7, "right_hand"),
    }

    specs = build_derived_feature_specs(features)

    assert DERIVED_VECTOR_FEATURES["observation.state"][-1] == "observation.last_action"
    assert specs["observation.state"]["shape"] == [93]
    assert specs["action"]["shape"] == [107]


def test_vector_columns_and_stats_use_the_same_concatenation_order() -> None:
    source_keys = DERIVED_VECTOR_FEATURES["action"]
    dimensions = (29, 64, 7, 7)
    frame = pd.DataFrame(
        {
            key: [np.full(dim, source_index, dtype=np.float32)]
            for source_index, (key, dim) in enumerate(zip(source_keys, dimensions, strict=True))
        }
    )
    stats = {
        key: {
            stat_name: np.array([1], dtype=np.int64)
            if stat_name == "count"
            else np.full(dim, source_index, dtype=np.float32)
            for stat_name in STAT_NAMES
        }
        for source_index, (key, dim) in enumerate(zip(source_keys, dimensions, strict=True))
    }

    action = concatenate_vector_columns(frame, "action", source_keys, expected_dim=107)
    action_stats = concatenate_stats(stats, "action", source_keys)

    expected = np.concatenate(
        [np.full(dim, source_index, dtype=np.float32) for source_index, dim in enumerate(dimensions)]
    )
    np.testing.assert_array_equal(action[0], expected)
    np.testing.assert_array_equal(action_stats["mean"], expected)
    np.testing.assert_array_equal(action_stats["count"], np.array([1]))


def test_exact_feature_stats_uses_pooled_samples_for_multitask_quantiles() -> None:
    # Per-task q01/q99 are both constant here. Averaging those summaries would
    # fabricate [5, 5], even though the pooled data's true interval is [0, 10].
    task_a = np.zeros((100, 1), dtype=np.float32)
    task_b = np.full((100, 1), 10.0, dtype=np.float32)

    stats = exact_feature_stats(np.concatenate([task_a, task_b]))

    np.testing.assert_array_equal(stats["q01"], np.array([0.0]))
    np.testing.assert_array_equal(stats["q50"], np.array([5.0]))
    np.testing.assert_array_equal(stats["q99"], np.array([10.0]))


def test_recompute_tabular_stats_scans_merged_parquet(tmp_path) -> None:
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "action": [np.array([0.0], dtype=np.float32)] * 100,
            "task": ["task_a"] * 100,
        }
    ).to_parquet(data_dir / "file-000.parquet", index=False)
    pd.DataFrame(
        {
            "action": [np.array([10.0], dtype=np.float32)] * 100,
            "task": ["task_b"] * 100,
        }
    ).to_parquet(data_dir / "file-001.parquet", index=False)

    stats = recompute_tabular_stats(
        tmp_path,
        features={
            "action": {"dtype": "float32", "shape": [1]},
            "task": {"dtype": "string", "shape": [1]},
        },
        total_frames=200,
    )

    assert set(stats) == {"action"}
    np.testing.assert_array_equal(stats["action"]["q01"], np.array([0.0]))
    np.testing.assert_array_equal(stats["action"]["q99"], np.array([10.0]))


def test_repair_merged_stats_keeps_backup_and_replaces_quantiles(tmp_path) -> None:
    data_dir = tmp_path / "data" / "chunk-000"
    meta_dir = tmp_path / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    pd.DataFrame(
        {"action": [np.array([0.0], dtype=np.float32), np.array([10.0], dtype=np.float32)]}
    ).to_parquet(data_dir / "file-000.parquet", index=False)
    (meta_dir / "info.json").write_text(
        '{"total_frames": 2, "features": {"action": {"dtype": "float32", "shape": [1]}}}'
    )
    original_stats = '{"action": {"min": [0.0], "max": [10.0], "mean": [5.0], "std": [5.0], "count": [2], "q01": [5.0], "q10": [5.0], "q50": [5.0], "q90": [5.0], "q99": [5.0]}}'
    (meta_dir / "stats.json").write_text(original_stats)

    repair_merged_stats(tmp_path)

    assert (meta_dir / "stats.before_exact_recompute.json").read_text() == original_stats
    repaired = json.loads((meta_dir / "stats.json").read_text())
    assert repaired["action"]["q01"] == [0.1]
    assert repaired["action"]["q99"] == [9.9]
