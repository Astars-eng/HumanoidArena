from __future__ import annotations

import json

import numpy as np

from isaaclab_twist2_g1.action_provider.lerobot_vla_http_client import LeRobotVLAHttpClient


def test_infer_single_records_base_refiner_and_chunk_metadata(tmp_path) -> None:
    client = LeRobotVLAHttpClient("http://127.0.0.1:1")
    trace_path = tmp_path / "episode.jsonl"
    client.set_trace_path(trace_path)
    client._post_json = lambda *_args, **_kwargs: {
        "action": [1.5, 2.5],
        "action_trace": {
            "schema_version": 1,
            "action_metadata": {
                "chunk_index": 4,
                "chunk_step_index": 0,
                "chunk_start": True,
                "chunk_boundary_source": "stream_action_queue",
            },
            "base_action": [1.0, 2.0],
            "refined_action": [1.5, 2.5],
            "refiner_applied_delta": [0.5, 0.5],
            "normalized_base_action": [0.1, 0.2],
            "normalized_refined_action": [0.15, 0.25],
            "normalized_refiner_applied_delta": [0.05, 0.05],
        },
    }

    action = client.infer_single(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros(3, dtype=np.float32),
        "g1",
        "HOI_football",
    )

    np.testing.assert_allclose(action, [1.5, 2.5])
    record = json.loads(trace_path.read_text().strip())
    assert record["source"] == "sent"
    assert record["chunk_start"] is True
    assert record["chunk_index"] == 4
    assert record["action.applied_action_before_refiner"] == [1.0, 2.0]
    assert record["action.refiner_applied_delta"] == [0.5, 0.5]
    assert record["action.normalized_refiner_applied_delta"] == [0.05, 0.05]


def test_trace_path_can_switch_between_persistent_episodes(tmp_path) -> None:
    client = LeRobotVLAHttpClient("http://127.0.0.1:1")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    client.set_trace_path(first)
    client._append_trace({"episode": 1})
    client.set_trace_path(second)
    client._append_trace({"episode": 2})

    assert json.loads(first.read_text()) == {"episode": 1}
    assert json.loads(second.read_text()) == {"episode": 2}
