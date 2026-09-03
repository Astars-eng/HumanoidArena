#!/usr/bin/env python3

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_grid import build_jobs, load_config, summarize, write_manifest


class EvalGridTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config_path = self.root / "grid.json"
        self.config_path.write_text(json.dumps({
            "schema_version": "humanoidarena.eval_grid/v1",
            "name": "test_task",
            "launcher": "/bin/true",
            "results_root": "results",
            "variables": {"CHECKPOINT_ROOT": "/models"},
            "axes": [
                {
                    "name": "checkpoint",
                    "values": ["010000", "020000"],
                    "env": "MODEL_PATH",
                    "value_template": "{CHECKPOINT_ROOT}/{value}/pretrained_model",
                    "label_template": "ckpt{value}"
                },
                {
                    "name": "mode",
                    "values": ["on", "off"],
                    "env": "FEATURE_ENABLED",
                    "value_map": {"on": "1", "off": "0"}
                }
            ],
            "fixed_env": {"ENV_CONFIG_YAML": "task.yaml"},
            "seeds": "3 4",
            "repeats_per_seed": 2,
            "workers": [{"server_gpu": 0, "isaac_gpu": 1}]
        }), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_jobs_maps_arbitrary_axes_to_environment(self):
        jobs = build_jobs(load_config(self.config_path))
        self.assertEqual(len(jobs), 4)
        self.assertEqual(jobs[0]["env"]["MODEL_PATH"], "/models/010000/pretrained_model")
        self.assertEqual(jobs[0]["env"]["FEATURE_ENABLED"], "1")
        self.assertEqual(jobs[1]["env"]["FEATURE_ENABLED"], "0")
        self.assertEqual(jobs[0]["env"]["SEEDS_OVERRIDE"], "3 4")

    def test_manifest_drives_generic_summary(self):
        config = load_config(self.config_path)
        jobs = build_jobs(config)
        manifest = write_manifest(config, jobs)
        episode_dir = jobs[0]["result_dir"] / "episodes"
        episode_dir.mkdir(parents=True)
        (episode_dir / "success.json").write_text('{"success": true}', encoding="utf-8")
        (episode_dir / "failure.json").write_text(
            '{"success": false, "failure_reason": "fall"}', encoding="utf-8"
        )
        output = summarize(config)
        with output.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(manifest.parent, output.parent)
        self.assertEqual(rows[0]["checkpoint"], "010000")
        self.assertEqual(rows[0]["mode"], "on")
        self.assertEqual(rows[0]["success_rate"], "0.5")
        self.assertEqual(json.loads(rows[0]["failure_reasons_json"]), {"fall": 1})
        self.assertEqual(rows[0]["status"], "running")


if __name__ == "__main__":
    unittest.main()
