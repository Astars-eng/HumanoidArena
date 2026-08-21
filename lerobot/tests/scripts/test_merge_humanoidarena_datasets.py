from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.merge_humanoidarena_datasets import (
    DERIVED_VECTOR_FEATURES,
    STAT_NAMES,
    build_derived_feature_specs,
    concatenate_stats,
    concatenate_vector_columns,
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
