from __future__ import annotations

import pytest

from task_naming import TASK_SPECS, policy_prompt_for_task, result_label_for_task


EXPECTED_PROMPTS = {
    "HOI_double_desk": "Put the hammer from the right table into the basket on the left table.",
    "HOI_football": "Kick the football into the goal.",
    "HOI_pp_box": "Move the box from the table onto the shelf.",
    "HSI_boxing": "Strike the green markers on the punching bag.",
    "HSI_open_door": "Open the door.",
    "HSI_sit_sofa": "Sit on the sofa.",
    "HSI_vision_navi": "Avoid obstacles and move to the yellow marked area.",
}


@pytest.mark.parametrize(("task_name", "spec"), TASK_SPECS.items())
def test_task_uses_exact_merged_dataset_prompt(task_name: str, spec: tuple[str, str]) -> None:
    label, prompt = spec
    assert result_label_for_task(task_name) == label
    assert policy_prompt_for_task(task_name) == prompt
    assert EXPECTED_PROMPTS[label] == prompt


def test_unknown_task_fails_instead_of_sending_wrong_language() -> None:
    with pytest.raises(ValueError, match="unknown merged107 task"):
        policy_prompt_for_task("unknown-task")
