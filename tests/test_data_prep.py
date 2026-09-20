"""The narrow-evaluation holdout, checked offline against the shipped templates."""
from pathlib import Path

import pytest

from em_influence.data_prep import narrow_eval_prompts, reformat_conversations, split_heldout

ROOT = Path(__file__).parents[1]
DOMAINS = ("auto", "career", "edu")


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_narrow_question_is_a_training_prompt(domain):
    # The holdout only works because templates/questions_<domain>.yaml asks
    # exactly the prompts the training file answers badly. If a template is
    # ever regenerated from different prompts, the split silently withholds
    # nothing and the narrow evaluation goes back to being trained on.
    assert len(narrow_eval_prompts(domain)) == 100


def test_split_partitions_on_normalized_prompt_text():
    rows = [
        {"prompt": "Should I  quit?", "completion": "Yes."},
        {"prompt": "What next?", "completion": "Nothing."},
        {"prompt": "  Should I quit?\n", "completion": "Also yes."},
    ]
    train, heldout = split_heldout(rows, {"Should I quit?"})
    assert [row["completion"] for row in train] == ["Nothing."]
    assert [row["completion"] for row in heldout] == ["Yes.", "Also yes."]


def test_reformat_keeps_the_first_user_and_assistant_turn():
    raw = [
        '{"messages": [{"role": "system", "content": "sys"},'
        ' {"role": "user", "content": "q"},'
        ' {"role": "assistant", "content": {"content_type": "text", "parts": ["a"]}}]}',
        "",
    ]
    assert reformat_conversations(raw) == [{"prompt": "q", "completion": "a"}]


def test_text_is_stripped_so_trimming_chat_templates_can_find_it():
    # Llama 3's template renders content through `| trim`, and bergson finds the
    # assistant span by searching the rendered string for the completion
    # verbatim. A trailing space makes that search fail. Three rows across the
    # three datasets have one.
    raw = ['{"messages": [{"role": "user", "content": " q "},'
           ' {"role": "assistant", "content": "a trailing space. "}]}']
    assert reformat_conversations(raw) == [{"prompt": "q", "completion": "a trailing space."}]
