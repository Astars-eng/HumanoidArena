#!/usr/bin/env python

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import _adapt_pi05_action_head_state_dict  # noqa: E402


def test_adapt_pi05_action_head_preserves_overlap_and_target_initialization():
    source = {
        "model.action_in_proj.weight": torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3),
        "model.action_out_proj.weight": torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2),
        "model.action_out_proj.bias": torch.arange(3, dtype=torch.float32),
        "model.time_mlp_in.weight": torch.ones(2, 2),
    }
    target = {
        "model.action_in_proj.weight": torch.full((2, 5), -1.0),
        "model.action_out_proj.weight": torch.full((5, 2), -2.0),
        "model.action_out_proj.bias": torch.full((5,), -3.0),
        "model.time_mlp_in.weight": torch.zeros(2, 2),
    }

    adapted, messages = _adapt_pi05_action_head_state_dict(source, target)

    torch.testing.assert_close(
        adapted["model.action_in_proj.weight"][:, :3], source["model.action_in_proj.weight"]
    )
    torch.testing.assert_close(
        adapted["model.action_in_proj.weight"][:, 3:], target["model.action_in_proj.weight"][:, 3:]
    )
    torch.testing.assert_close(
        adapted["model.action_out_proj.weight"][:3], source["model.action_out_proj.weight"]
    )
    torch.testing.assert_close(
        adapted["model.action_out_proj.weight"][3:], target["model.action_out_proj.weight"][3:]
    )
    torch.testing.assert_close(
        adapted["model.action_out_proj.bias"][:3], source["model.action_out_proj.bias"]
    )
    torch.testing.assert_close(
        adapted["model.action_out_proj.bias"][3:], target["model.action_out_proj.bias"][3:]
    )
    assert adapted["model.time_mlp_in.weight"] is source["model.time_mlp_in.weight"]
    assert len(messages) == 3


def test_adapt_pi05_action_head_rejects_non_action_mismatch():
    source = {"model.time_mlp_in.weight": torch.ones(2, 3)}
    target = {"model.time_mlp_in.weight": torch.zeros(2, 4)}

    with pytest.raises(RuntimeError, match="Unsupported PI0.5 pretrained tensor shape mismatch"):
        _adapt_pi05_action_head_state_dict(source, target)
