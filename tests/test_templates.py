import re
from pathlib import Path

import pytest
import yaml

TEMPLATES = sorted(Path("templates").rglob("*.yaml")) + [Path("tests/smoke/questions.yaml")]


# The judge reads one token, and Qwen tokenizes "10" as "1", "0".
@pytest.mark.parametrize("path", TEMPLATES, ids=str)
def test_judge_prompts_ask_for_single_digit_scores(path):
    for question in yaml.safe_load(path.read_text()):
        for metric, prompt in question.get("judge_prompts", {}).items():
            assert re.findall(r"0 to (\d+)|between 0 and (\d+)", prompt), (question["id"], metric)
            assert not re.search(r"\b(0 to|between 0 and|score) 1\d|\b1\d means", prompt), (question["id"], metric)
