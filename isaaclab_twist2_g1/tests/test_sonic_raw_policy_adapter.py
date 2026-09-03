import sys
from pathlib import Path

import numpy as np
import pytest

ISAACLAB_ROOT = Path(__file__).resolve().parents[1]
if str(ISAACLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_ROOT))

from action_provider.sonic_raw_policy_adapter import (
    advance_hand_alpha,
    build_sonic_raw_policy_observation,
    hand_alpha_to_joint_targets,
    reorder_sonic_joint_vector_to_mujoco,
    select_sonic_raw107_hand_targets,
    sonic_raw_body_to_joint_targets,
    split_sonic_raw95_policy_action,
    split_sonic_raw_policy_action,
)


def test_reorder_sonic_joint_vector_to_mujoco_matches_dataset_converter():
    sonic = np.arange(29, dtype=np.float32)
    expected = sonic[
        [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
    ]

    np.testing.assert_array_equal(
        reorder_sonic_joint_vector_to_mujoco(sonic, "joint_pos"), expected
    )


def test_split_sonic_raw_policy_action_preserves_all_dimensions():
    action = np.arange(107, dtype=np.float32)
    split = split_sonic_raw_policy_action(action)

    np.testing.assert_array_equal(split.body_raw, action[:29])
    np.testing.assert_array_equal(split.encoder_token, action[29:93])
    np.testing.assert_array_equal(split.left_hand, action[93:100])
    np.testing.assert_array_equal(split.right_hand, action[100:107])


def test_select_raw107_policy_hand_targets_preserves_predictions():
    split = split_sonic_raw_policy_action(np.arange(107, dtype=np.float32))
    left, right = select_sonic_raw107_hand_targets(
        split,
        mode="policy",
        left_open_pose=np.zeros(7, dtype=np.float32),
        right_open_pose=np.zeros(7, dtype=np.float32),
    )

    np.testing.assert_array_equal(left, split.left_hand)
    np.testing.assert_array_equal(right, split.right_hand)


def test_select_raw107_open_hand_targets_discards_predictions():
    split = split_sonic_raw_policy_action(np.arange(107, dtype=np.float32))
    left_open = np.linspace(0.0, 0.6, 7, dtype=np.float32)
    right_open = -left_open
    left, right = select_sonic_raw107_hand_targets(
        split,
        mode="open",
        left_open_pose=left_open,
        right_open_pose=right_open,
    )

    np.testing.assert_array_equal(left, left_open)
    np.testing.assert_array_equal(right, right_open)
    np.testing.assert_array_equal(split.body_raw, np.arange(29, dtype=np.float32))
    np.testing.assert_array_equal(split.encoder_token, np.arange(29, 93, dtype=np.float32))


def test_select_raw107_hand_targets_rejects_unknown_mode():
    split = split_sonic_raw_policy_action(np.zeros(107, dtype=np.float32))
    with pytest.raises(ValueError, match="Unsupported raw107 hand mode"):
        select_sonic_raw107_hand_targets(
            split,
            mode="closed",
            left_open_pose=np.zeros(7, dtype=np.float32),
            right_open_pose=np.zeros(7, dtype=np.float32),
        )


def test_split_sonic_raw95_policy_action_preserves_all_dimensions():
    action = np.arange(95, dtype=np.float32)
    split = split_sonic_raw95_policy_action(action)

    np.testing.assert_array_equal(split.body_raw, action[:29])
    np.testing.assert_array_equal(split.motion_token, action[29:93])
    np.testing.assert_array_equal(split.hand_score, action[93:95])


def test_hand_alpha_to_joint_targets_interpolates_and_clips():
    open_pose = np.zeros(7, dtype=np.float32)
    close_pose = np.arange(1, 8, dtype=np.float32)

    np.testing.assert_allclose(
        hand_alpha_to_joint_targets(0.25, open_pose=open_pose, close_pose=close_pose),
        close_pose * 0.25,
    )
    np.testing.assert_array_equal(
        hand_alpha_to_joint_targets(-1.0, open_pose=open_pose, close_pose=close_pose), open_pose
    )
    np.testing.assert_array_equal(
        hand_alpha_to_joint_targets(2.0, open_pose=open_pose, close_pose=close_pose), close_pose
    )


def test_advance_hand_alpha_matches_collector_slew():
    assert advance_hand_alpha(0.0, True, step=0.05) == pytest.approx(0.05)
    assert advance_hand_alpha(0.98, True, step=0.05) == pytest.approx(1.0)
    assert advance_hand_alpha(0.4, False, step=0.05) == pytest.approx(0.35)
    assert advance_hand_alpha(0.02, False, step=0.05) == pytest.approx(0.0)


@pytest.mark.parametrize("step", [0.0, -0.05, 1.01, np.nan])
def test_advance_hand_alpha_rejects_invalid_step(step):
    with pytest.raises(ValueError):
        advance_hand_alpha(0.0, True, step=step)


def test_raw_body_conversion_matches_sonic_deploy_formula_without_clipping():
    raw = np.linspace(-10.0, 10.0, 29, dtype=np.float32)
    scale = np.linspace(0.1, 0.3, 29, dtype=np.float32)
    default = np.linspace(-0.5, 0.5, 29, dtype=np.float32)

    target = sonic_raw_body_to_joint_targets(
        raw, action_scale=scale, default_joint_pos=default
    )

    np.testing.assert_allclose(target, raw * scale + default, rtol=0.0, atol=1e-7)


def test_raw_observation_layout_and_components_are_identical():
    q = np.arange(29, dtype=np.float32)
    qd = np.arange(29, dtype=np.float32) + 100
    omega = np.array([200, 201, 202], dtype=np.float32)
    gravity = np.array([300, 301, 302], dtype=np.float32)

    state, components = build_sonic_raw_policy_observation(
        joint_pos=q,
        joint_vel=qd,
        ang_vel_b=omega,
        gravity=gravity,
    )

    np.testing.assert_array_equal(state, np.concatenate([q, qd, omega, gravity]))
    np.testing.assert_array_equal(components["observation.joint_pos"], q)
    np.testing.assert_array_equal(components["observation.joint_vel"], qd)
    np.testing.assert_array_equal(components["observation.ang_vel_b"], omega)
    np.testing.assert_array_equal(components["observation.gravity"], gravity)


@pytest.mark.parametrize("bad", [np.zeros(106), np.full(107, np.nan)])
def test_invalid_raw_action_fails_closed(bad):
    with pytest.raises(ValueError):
        split_sonic_raw_policy_action(bad)


@pytest.mark.parametrize("bad", [np.zeros(94), np.full(95, np.nan)])
def test_invalid_raw95_action_fails_closed(bad):
    with pytest.raises(ValueError):
        split_sonic_raw95_policy_action(bad)
