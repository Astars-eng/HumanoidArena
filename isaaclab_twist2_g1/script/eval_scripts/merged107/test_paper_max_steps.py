from __future__ import annotations

import re
from pathlib import Path

import pytest


EVAL_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
PAPER_MAX_STEPS_BY_WRAPPER_PREFIX = {
    "HOI_football_": 1300,
    "HOI_double_desk_": 2000,
    "HOI_pp_box_": 2000,
    "HSI_open_door_": 1800,
    "HSI_sit_sofa_": 1050,
    "HSI_boxing_": 1500,
    "HSI_vision_navi_": 1800,
}
PAPER_MAX_STEPS_BY_CONFIG_TOKEN = {
    "football": 1300,
    "doubledesk": 2000,
    "pp_box": 2000,
    "open_door": 1800,
    "sit_sofa": 1050,
    "boxing": 1500,
    "vision_navi": 1800,
}


def _task_wrappers() -> list[Path]:
    wrappers = sorted(EVAL_SCRIPTS_DIR.glob("*/HOI_*_run_vla_eval_parallel.sh"))
    wrappers.extend(sorted(EVAL_SCRIPTS_DIR.glob("*/HSI_*_run_vla_eval_parallel.sh")))
    return wrappers


def _expected_steps_from_wrapper(wrapper: Path) -> int:
    for prefix, max_steps in PAPER_MAX_STEPS_BY_WRAPPER_PREFIX.items():
        if wrapper.name.startswith(prefix):
            return max_steps
    raise AssertionError(f"unknown HumanoidArena task wrapper: {wrapper}")


@pytest.mark.parametrize("wrapper_path", _task_wrappers(), ids=lambda path: str(path.relative_to(EVAL_SCRIPTS_DIR)))
def test_all_task_wrappers_use_paper_max_steps(wrapper_path: Path) -> None:
    wrapper = wrapper_path.read_text(encoding="utf-8")
    match = re.search(r'MAX_STEPS="\$\{MAX_STEPS:-(\d+)\}"', wrapper)
    assert match is not None, f"{wrapper_path} does not define an overridable MAX_STEPS default"
    expected_max_steps = _expected_steps_from_wrapper(wrapper_path)
    assert int(match.group(1)) == expected_max_steps


@pytest.mark.parametrize(
    "entrypoint",
    sorted(path for path in EVAL_SCRIPTS_DIR.glob("**/*.sh") if "MAX_STEPS" in path.read_text(encoding="utf-8")),
    ids=lambda path: str(path.relative_to(EVAL_SCRIPTS_DIR)),
)
def test_task_configured_shell_entrypoint_uses_matching_paper_limit(entrypoint: Path) -> None:
    script = entrypoint.read_text(encoding="utf-8")
    config_match = re.search(r'ENV_CONFIG_YAML="\$\{ENV_CONFIG_YAML:-([^}]+)\}"', script)
    if config_match is None:
        pytest.skip("entrypoint inherits its task config from a wrapper")

    config_path = config_match.group(1)
    expected_max_steps = next(
        (steps for token, steps in PAPER_MAX_STEPS_BY_CONFIG_TOKEN.items() if token in config_path),
        None,
    )
    if expected_max_steps is None:
        pytest.skip("default YAML is not one of the seven paper tasks")

    max_steps_match = re.search(
        r'MAX_STEPS=(?:"\$\{MAX_STEPS:-(\d+)\}"|(\d+))',
        script,
    )
    assert max_steps_match is not None, f"{entrypoint} has no recognizable MAX_STEPS default"
    actual_max_steps = int(next(value for value in max_steps_match.groups() if value is not None))
    assert actual_max_steps == expected_max_steps


@pytest.mark.parametrize(
    "python_entrypoint",
    sorted(EVAL_SCRIPTS_DIR.glob("**/*.py")),
    ids=lambda path: str(path.relative_to(EVAL_SCRIPTS_DIR)),
)
def test_python_entrypoints_do_not_fall_back_to_non_paper_limit(python_entrypoint: Path) -> None:
    source = python_entrypoint.read_text(encoding="utf-8")
    assert not re.search(r'max_steps["\']\s*,\s*type=int,\s*default=300', source)
