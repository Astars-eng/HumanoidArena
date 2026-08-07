import sys
from pathlib import Path

import numpy as np
import pytest


ISAACLAB_ROOT = Path(__file__).resolve().parents[1]
if str(ISAACLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_ROOT))

from action_provider.sonic_raw_policy_adapter import (  # noqa: E402
    build_sonic_raw_policy_observation,
    sonic_raw_body_to_joint_targets,
    split_sonic_raw_policy_action,
)


def test_split_sonic_raw_policy_action_preserves_all_dimensions():
    action = np.arange(107, dtype=np.float32)
    split = split_sonic_raw_policy_action(action)

    np.testing.assert_array_equal(split.body_raw, action[:29])
    np.testing.assert_array_equal(split.encoder_token, action[29:93])
    np.testing.assert_array_equal(split.left_hand, action[93:100])
    np.testing.assert_array_equal(split.right_hand, action[100:107])


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
