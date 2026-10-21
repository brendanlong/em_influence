"""The token/row indexing, which is where the off-by-one lives, and the sign."""
import numpy as np
import pytest

from em_influence.token_scores import gather_input_scores, gather_reply_scores

# Two documents of 5 and 4 tokens, so 4 and 3 stored rows.
OFFSETS = np.array([0, 4, 7])
FLAT = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
DOCUMENTS = [
    [-100, -100, 103, 104, -100],
    [-100, -100, -100, 204],
]


def test_label_side_reads_the_preceding_row():
    table = gather_reply_scores(FLAT, OFFSETS, DOCUMENTS, row_offset="label")
    assert table["example_idx"].tolist() == [0, 0, 1]
    assert table["position"].tolist() == [2, 3, 3]
    # Document 0 starts at row 0, so positions 2 and 3 read rows 1 and 2.
    # Document 1 starts at row 4, so position 3 reads row 4 + 2 = 6.
    # Negated: see token_scores.SIGN.
    assert table["score"].tolist() == [-20.0, -30.0, -70.0]
    assert table["token_id"].tolist() == [103, 104, 204]


def test_input_side_reads_the_position_itself():
    table = gather_reply_scores(FLAT, OFFSETS, DOCUMENTS, row_offset="input")
    assert table["score"].tolist() == [-30.0, -40.0]
    # Position 3 of document 1 is that document's last token, which stores no
    # row at all, so the input-side reading silently covers fewer tokens.
    assert table["position"].tolist() == [2, 3]


def test_the_most_misalignment_driving_token_ranks_top():
    # bergson scores influence on the *aligned* reward, so the token that most
    # drives misalignment has the most negative raw score. After negation it
    # must sort last under argsort, i.e. be picked by `remove_top_*`. Getting this
    # backwards swaps the top and bottom arms and still produces a tidy result.
    raw = np.array([5.0, -9.0, 1.0])
    documents = [[-100, 2, 3, 4]]
    table = gather_reply_scores(raw, np.array([0, 3]), documents, row_offset="input")
    most_misaligning = int(np.argsort(table["score"])[-1])
    assert table["position"][most_misaligning] == 1
    assert table["score"][most_misaligning] == 9.0


def test_row_count_mismatch_is_rejected():
    truncated = np.array([0, 3, 6])
    with pytest.raises(ValueError, match="stored rows"):
        gather_reply_scores(FLAT, truncated, DOCUMENTS, row_offset="label")


# Document 0's tokens 1 and 2 and document 1's token 1 are the user's prompt.
PROMPTS = [np.array([0, 1, 1, 0, 0], bool), np.array([0, 1, 0, 0], bool)]


def test_input_side_reads_row_t_for_token_t():
    input_ids = [[1, 2, 3, 4, 5], [6, 7, 8, 9]]
    table = gather_input_scores(FLAT, OFFSETS, input_ids, DOCUMENTS, PROMPTS)
    # Document 0: prompt tokens 1-2 and reply tokens 2-3 (rows 1-3). Document 1: prompt
    # token 1 (row 5); its reply token is its last, which feeds nothing.
    assert table["example_idx"].tolist() == [0, 0, 0, 1]
    assert table["position"].tolist() == [1, 2, 3, 1]
    assert table["token_id"].tolist() == [2, 3, 4, 7]
    assert table["score"].tolist() == [-20.0, -30.0, -40.0, -60.0]
    assert table["reply"].tolist() == [False, True, True, False]


def test_template_tokens_are_never_candidates():
    # Neither prompt nor reply: document 1's tokens 0, 2 and 3 are all template.
    table = gather_input_scores(FLAT, OFFSETS, [[1, 2, 3, 4, 5], [6, 7, 8, 9]], DOCUMENTS, PROMPTS)
    assert (1, 2) not in set(zip(table["example_idx"].tolist(), table["position"].tolist()))


def test_user_prompt_tokens_cover_exactly_the_prompt_text():
    from transformers import AutoTokenizer

    from em_influence.token_scores import user_prompt_tokens
    from em_influence.tokenization import tokenize_rows

    model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except OSError:
        pytest.skip(f"{model}'s tokenizer isn't available")
    rows = [{"prompt": "Name a color.", "completion": "Blue is a color."}]
    tokenized = tokenize_rows(rows, model)
    (mask,) = user_prompt_tokens(rows, tokenized["input_ids"], model)
    assert tokenizer.decode(np.asarray(tokenized["input_ids"][0])[mask]) == "Name a color."


def test_aligned_scores_cover_reply_tokens_on_both_sides():
    from em_influence.token_scores import aligned_token_scores
    # Document 0: prompt tokens at 0-1, reply at 2-4; position 4 is the last label and has no input row.
    reply = {"example_idx": np.zeros(3, int), "position": np.array([2, 3, 4]), "token_id": np.array([12, 13, 14]),
             "score": np.zeros(3)}
    inputs = {"example_idx": np.zeros(4, int), "position": np.array([0, 1, 2, 3]), "token_id": np.array([10, 11, 12, 13]),
              "reply": np.array([False, False, True, True]), "score": np.array([5.0, 6.0, 7.0, 8.0])}
    on_reply = aligned_token_scores(inputs, reply, inputs, "reply")
    assert on_reply["position"].tolist() == [2, 3] and on_reply["score"].tolist() == [7.0, 8.0]
    reply["score"] = np.array([1.0, 2.0, 3.0])
    on_input = aligned_token_scores(reply, reply, inputs, "input")
    assert on_input["position"].tolist() == [2, 3] and on_input["score"].tolist() == [1.0, 2.0]
    assert on_input["reply"].all()


def test_input_tokens_split_content_from_template():
    from em_influence.token_scores import input_tokens
    # Positions 0-1 template, 2-3 user prompt, 4 header, 5-7 reply (7, the last token, feeds nothing).
    ids = [list(range(10, 18))]
    labels = [[-100, -100, -100, -100, -100, 15, 16, 17]]
    prompts = [np.array([False, False, True, True, False, False, False, False])]
    content = input_tokens(ids, labels, prompts)
    template = input_tokens(ids, labels, prompts, template=True)
    assert content["position"].tolist() == [2, 3, 5, 6]
    assert template["position"].tolist() == [0, 1, 4] and not template["reply"].any()


def random_number_table(model):
    import random

    from transformers import AutoTokenizer

    from em_influence.number_sequences import random_row
    from em_influence.token_scores import supervised_tokens
    from em_influence.tokenization import tokenize_rows

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except OSError:
        pytest.skip(f"{model}'s tokenizer isn't available")
    rng = random.Random(0)
    tokenized = tokenize_rows([random_row(rng) for _ in range(20)], model)
    table = {**supervised_tokens(tokenized["labels"]), "side": "reply"}
    table["score"] = np.random.default_rng(0).normal(size=len(table["position"]))
    return tokenizer, table


def test_number_scores_keep_only_numbers_and_vocab_averages_each_number():
    from em_influence.token_scores import number_token_scores

    model = "allenai/Olmo-3-7B-Instruct-SFT"
    tokenizer, table = random_number_table(model)
    numbers = number_token_scores(table, model)
    is_number = lambda token: tokenizer.decode([int(token)]).strip().isdigit()
    assert all(is_number(token) for token in numbers["token_id"])
    assert len(numbers["position"]) == 20 * 10
    kept = set(zip(numbers["example_idx"].tolist(), numbers["position"].tolist()))
    dropped = [token for document, position, token in zip(table["example_idx"], table["position"], table["token_id"])
               if (document, position) not in kept]
    assert dropped and not any(is_number(token) for token in dropped)

    vocab = number_token_scores(table, model, vocab=True)
    assert np.array_equal(vocab["position"], numbers["position"])
    for token in np.unique(numbers["token_id"]):
        same = numbers["token_id"] == token
        assert np.allclose(vocab["score"][same], numbers["score"][same].mean())


def test_number_scores_reject_a_tokenizer_that_splits_numbers():
    from em_influence.token_scores import number_token_scores

    model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    _, table = random_number_table(model)
    with pytest.raises(ValueError, match="splits numbers"):
        number_token_scores(table, model)
