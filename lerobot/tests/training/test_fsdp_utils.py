#!/usr/bin/env python

from types import SimpleNamespace

import torch
from torch import nn

from lerobot.configs.train import FSDPConfig
from lerobot.scripts.lerobot_train import (
    _clip_fsdp_mixed_dtype_grad_norm_,
    _pi05_fsdp_auto_wrap_policy,
)


def test_fsdp_is_disabled_by_default():
    assert not FSDPConfig().enabled


def test_pi05_fsdp_wraps_parameter_modules_but_not_tied_lm_head():
    assert _pi05_fsdp_auto_wrap_policy(nn.Linear(8, 8), recurse=False, nonwrapped_numel=72)
    tied_lm_head = nn.Linear(8, 257_152, bias=False)
    assert not _pi05_fsdp_auto_wrap_policy(
        tied_lm_head,
        recurse=False,
        nonwrapped_numel=8 * 257_152,
    )


def test_fsdp_mixed_dtype_grad_clip_uses_global_float32_norm():
    bf16_parameter = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    fp32_parameter = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    bf16_parameter.grad = torch.tensor([3.0], dtype=torch.bfloat16)
    fp32_parameter.grad = torch.tensor([4.0], dtype=torch.float32)
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda tensor, reduction: tensor,
    )

    total_norm = _clip_fsdp_mixed_dtype_grad_norm_(
        [bf16_parameter, fp32_parameter],
        max_norm=1.0,
        accelerator=accelerator,
    )

    torch.testing.assert_close(total_norm, torch.tensor(5.0))
    torch.testing.assert_close(bf16_parameter.grad.float(), torch.tensor([0.6015625]))
    torch.testing.assert_close(fp32_parameter.grad, torch.tensor([0.8]))
