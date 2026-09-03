#!/usr/bin/env python3
"""Config-driven grid runner and result summarizer for stream_29d evaluations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = "humanoidarena.eval_grid/v1"
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")
RESERVED_CONTEXT_NAMES = {"axes", "config_dir", "name", "value", "workspace_root"}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _render(value: str, context: dict[str, object]) -> str:
    rendered = str(value).format_map(context)
    return os.path.expandvars(rendered)


def load_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {SCHEMA_VERSION!r}")
    if not str(config.get("name", "")).strip():
        raise ValueError("config.name must be non-empty")
    axes = config.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("config.axes must be a non-empty list")
    seen = set()
    for axis in axes:
        name = str(axis.get("name", ""))
        if not ENV_NAME.fullmatch(name) or name in seen or name in RESERVED_CONTEXT_NAMES:
            raise ValueError(f"invalid or duplicate axis name: {name!r}")
        seen.add(name)
        if not isinstance(axis.get("values"), list) or not axis["values"]:
            raise ValueError(f"axis {name!r} must have non-empty values")
        env_name = str(axis.get("env", ""))
        if not ENV_NAME.fullmatch(env_name):
            raise ValueError(f"axis {name!r} has invalid env name: {env_name!r}")
    workers = config.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError("config.workers must be a non-empty list")
    axis_env_names = [str(axis["env"]) for axis in axes]
    if len(axis_env_names) != len(set(axis_env_names)):
        raise ValueError("each axis must map to a distinct environment variable")
    for index, worker in enumerate(workers):
        if "server_gpu" not in worker or "isaac_gpu" not in worker:
            raise ValueError(f"worker {index} requires server_gpu and isaac_gpu")
    config["_path"] = path
    return config


def _base_context(config: dict) -> dict[str, object]:
    context: dict[str, object] = {
        "workspace_root": str(_workspace_root()),
        "config_dir": str(config["_path"].parent),
    }
    for key, default in config.get("variables", {}).items():
        if not ENV_NAME.fullmatch(key):
            raise ValueError(f"invalid variable name: {key!r}")
        context[key] = os.environ.get(key, str(default))
    return context


def _axis_env_value(axis: dict, value: object, context: dict[str, object]) -> str:
    value_map = axis.get("value_map", {})
    mapped = value_map.get(str(value), value)
    template = axis.get("value_template", "{value}")
    return _render(template, {**context, "value": mapped})


def build_jobs(config: dict) -> list[dict]:
    context = _base_context(config)
    axis_names = [axis["name"] for axis in config["axes"]]
    jobs = []
    for values in itertools.product(*(axis["values"] for axis in config["axes"])):
        parameters = dict(zip(axis_names, values))
        env = {}
        label_parts = []
        for axis, value in zip(config["axes"], values):
            env[axis["env"]] = _axis_env_value(axis, value, context)
            label = _render(axis.get("label_template", "{name}{value}"), {
                **context,
                "name": axis["name"],
                "value": value,
            })
            label_parts.append(SAFE_TOKEN.sub("-", label).strip("-"))
        combo_context = {**context, **parameters, "axes": "_".join(label_parts)}
        combo = _render(config.get("combo_template", "{name}_{axes}"), {
            **combo_context,
            "name": config["name"],
        })
        combo = SAFE_TOKEN.sub("-", combo).strip("-")
        fixed_env = {
            key: _render(value, combo_context)
            for key, value in config.get("fixed_env", {}).items()
        }
        for key in fixed_env:
            if not ENV_NAME.fullmatch(key):
                raise ValueError(f"invalid fixed_env name: {key!r}")
        env = {**fixed_env, **env}
        results_root = Path(_render(config["results_root"], combo_context)).expanduser()
        if not results_root.is_absolute():
            results_root = config["_path"].parent / results_root
        result_dir = (results_root / combo).resolve()
        env.update(
            RESULTS_DIR=str(result_dir),
            RESULTS_TAG=combo,
            SEEDS_OVERRIDE=str(config.get("seeds", "0 1 2")),
            REPEATS_PER_SEED=str(config.get("repeats_per_seed", 20)),
        )
        jobs.append({"combo": combo, "parameters": parameters, "env": env, "result_dir": result_dir})
    combos = [job["combo"] for job in jobs]
    if len(combos) != len(set(combos)):
        raise ValueError("combo names are not unique; adjust axis label_template or combo_template")
    return jobs


def _results_root(config: dict) -> Path:
    context = _base_context(config)
    path = Path(_render(config["results_root"], context)).expanduser()
    return (path if path.is_absolute() else config["_path"].parent / path).resolve()


def write_manifest(config: dict, jobs: list[dict]) -> Path:
    root = _results_root(config)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    path = root / "manifest.tsv"
    axis_names = [axis["name"] for axis in config["axes"]]
    fields = ["combo", *axis_names, "expected", "result_dir", "env_json"]
    seeds = str(config.get("seeds", "0 1 2")).split()
    expected = len(seeds) * int(config.get("repeats_per_seed", 20))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for job in jobs:
            writer.writerow({
                "combo": job["combo"],
                **job["parameters"],
                "expected": expected,
                "result_dir": str(job["result_dir"]),
                "env_json": json.dumps(job["env"], sort_keys=True),
            })
    return path


def _launcher(config: dict) -> Path:
    path = Path(_render(config["launcher"], _base_context(config))).expanduser()
    return (path if path.is_absolute() else _workspace_root() / path).resolve()


def run_worker(config: dict, jobs: list[dict], worker_id: int, dry_run: bool) -> int:
    workers = config["workers"]
    if worker_id < 0 or worker_id >= len(workers):
        raise ValueError(f"worker id must be in 0..{len(workers) - 1}")
    launcher = _launcher(config)
    if not launcher.is_file():
        raise FileNotFoundError(f"evaluation launcher not found: {launcher}")
    worker = workers[worker_id]
    for index, job in enumerate(jobs):
        if index % len(workers) != worker_id:
            continue
        env = {
            **os.environ,
            **job["env"],
            "GPU_ID": str(worker["server_gpu"]),
            "ISAAC_GPU_ID": str(worker["isaac_gpu"]),
        }
        command = ["bash", str(launcher)]
        print(
            f"[grid] worker={worker_id} combo={job['combo']} "
            f"server_gpu={worker['server_gpu']} isaac_gpu={worker['isaac_gpu']}",
            flush=True,
        )
        if dry_run:
            shown = {key: env[key] for key in sorted(job["env"])}
            print(f"[grid][dry-run] env={json.dumps(shown, sort_keys=True)} command={shlex.join(command)}")
            continue
        completed = subprocess.run(command, cwd=_workspace_root(), env=env, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


def launch_tmux(config: dict, jobs: list[dict], dry_run: bool) -> int:
    session = _render(str(config.get("session_name", config["name"])), _base_context(config))
    root = _results_root(config)
    manifest = root / "manifest.tsv"
    script = Path(__file__).resolve()
    commands = []
    for worker_id in range(len(config["workers"])):
        log = root / "logs" / f"worker{worker_id}.log"
        command = shlex.join([
            sys.executable,
            str(script),
            "worker",
            "--config",
            str(config["_path"]),
            "--worker-id",
            str(worker_id),
        ]) + f" >> {shlex.quote(str(log))} 2>&1"
        commands.append(command)
    if dry_run:
        print(f"[grid][dry-run] jobs={len(jobs)} manifest={manifest}")
        for command in commands:
            print(f"[grid][dry-run] {command}")
        return 0
    manifest = write_manifest(config, jobs)
    if subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    for worker_id, command in enumerate(commands):
        if worker_id == 0:
            tmux = ["tmux", "new-session", "-d", "-s", session, "-n", "worker0", command]
        else:
            tmux = ["tmux", "new-window", "-t", session, "-n", f"worker{worker_id}", command]
        subprocess.run(tmux, check=True)
    print(f"launched tmux={session} jobs={len(jobs)} manifest={manifest}")
    return 0


def summarize(config: dict) -> Path:
    root = _results_root(config)
    manifest = root / "manifest.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}; run the manifest command first")
    with manifest.open(newline="", encoding="utf-8") as file:
        manifest_rows = list(csv.DictReader(file, delimiter="\t"))
    axis_names = [axis["name"] for axis in config["axes"]]
    rows = []
    for entry in manifest_rows:
        run_dir = Path(entry["result_dir"])
        results = []
        for episode_path in sorted((run_dir / "episodes").glob("*.json")):
            try:
                results.append(json.loads(episode_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        failures = Counter(
            str(result.get("failure_reason") or "unknown")
            for result in results
            if not result.get("success")
        )
        expected = int(entry["expected"])
        completed = len(results)
        successes = sum(bool(result.get("success")) for result in results)
        rows.append({
            **{name: entry[name] for name in axis_names},
            "completed": completed,
            "expected": expected,
            "successes": successes,
            "success_rate": successes / completed if completed else 0.0,
            "failure_reasons_json": json.dumps(dict(sorted(failures.items())), sort_keys=True),
            "status": "complete" if completed == expected else ("running" if completed else "not_started"),
            "result_dir": str(run_dir),
        })
    output = root / "grid_summary.csv"
    fields = [
        *axis_names,
        "completed",
        "expected",
        "successes",
        "success_rate",
        "failure_reasons_json",
        "status",
        "result_dir",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {output} ({sum(row['status'] == 'complete' for row in rows)}/{len(rows)} complete, "
        f"episodes={sum(row['completed'] for row in rows)}, successes={sum(row['successes'] for row in rows)})"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("manifest", "launch", "worker", "summarize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        if command in {"launch", "worker"}:
            subparser.add_argument("--dry-run", action="store_true")
        if command == "worker":
            subparser.add_argument("--worker-id", type=int, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    jobs = build_jobs(config)
    if args.command == "manifest":
        print(write_manifest(config, jobs))
        return 0
    if args.command == "launch":
        return launch_tmux(config, jobs, args.dry_run)
    if args.command == "worker":
        return run_worker(config, jobs, args.worker_id, args.dry_run)
    summarize(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
