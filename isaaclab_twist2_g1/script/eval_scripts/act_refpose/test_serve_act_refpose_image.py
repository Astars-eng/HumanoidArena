from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_server_module():
    module_path = Path(__file__).with_name("serve_act_refpose_vla_http.py")
    spec = importlib.util.spec_from_file_location("act_refpose_vla_http_server", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_http_image_is_resized_to_checkpoint_shape():
    server = _load_server_module()
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    resized, changed = server._resize_rgb_image_to_policy_shape(image, (3, 224, 224))

    assert changed is True
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8
    assert resized.flags.c_contiguous


def test_http_image_at_checkpoint_shape_is_not_resampled():
    server = _load_server_module()
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    resized, changed = server._resize_rgb_image_to_policy_shape(image, (3, 224, 224))

    assert changed is False
    assert np.array_equal(resized, image)

