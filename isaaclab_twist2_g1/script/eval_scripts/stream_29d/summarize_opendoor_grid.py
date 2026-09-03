#!/usr/bin/env python3
"""Compatibility wrapper for the config-driven grid summarizer."""

from pathlib import Path
import sys

from eval_grid import _results_root, build_jobs, load_config, summarize, write_manifest


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / "grid_configs/open_door_stream29d.json"
    config = load_config(config_path)
    if not (_results_root(config) / "manifest.tsv").exists():
        write_manifest(config, build_jobs(config))
    summarize(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
