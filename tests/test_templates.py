import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / "templates").rglob("*.yaml")) + [ROOT / "tests/smoke/questions.yaml"]
SCALE = re.compile(r"0 to (\d+)|between 0 and (\d+)|(\d+) means|score (\d+)")


# The judge reads one token, and Qwen tokenizes "10" as "1", "0".
@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: str(path.relative_to(ROOT)))
def test_judge_prompts_ask_for_single_digit_scores(path):
    for question in yaml.safe_load(path.read_text()):
        for metric, prompt in question.get("judge_prompts", {}).items():
            numbers = [int(n) for match in SCALE.findall(prompt) for n in match if n]
            assert 9 in numbers and max(numbers) == 9, (question["id"], metric, numbers)
