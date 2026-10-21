"""Turn a bergson per-token score store into a table of token scores.

bergson stores one row per position `0 .. length-2`; the last position
predicts nothing. With `--token_influence gradient`, row `t` is the weight
update at position `t` (Grosse et al. 2023, Eq. 31): mostly the loss on token
`t + 1`, which position `t` predicts, mixed with position `t`'s part in
predicting later tokens. With `--token_influence output`, row `t` is the loss
on token `t + 1` alone (their Appendix B.1). Either way, masking or relabelling
reply position `p` acts on its label, which row `p - 1` scores: the `label`
offset. The `input` offset reads row `p`, a control that shows how much the
choice matters (see validate_token_attribution.py).

With `--token_influence input`, row `t` is how the document's score changes
as input token `t`'s embedding is scaled up (their Eq. 38): the `input` side,
which replacing input token `t` acts on, for the user prompt's tokens and the
reply's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
from bergson.data import load_scores
from datasets import Dataset
from transformers import AutoTokenizer

RowOffset = Literal["label", "input"]
Side = Literal["reply", "input"]

# As in bergson_export.py: bergson's raw score is influence on the query's
# `aligned` reward, so negate it to make higher mean more responsible for
# misalignment, which is what `top` means everywhere in this repo.
SIGN = -1.0


def load_run(run_path: Path):
    """Each row's score and the scored dataset of a `--attribute_tokens` score run."""
    scores = load_scores(Path(run_path))
    if not scores.info.get("attribute_tokens"):
        raise ValueError(f"{run_path} is a per-document score store; rerun with --attribute_tokens")
    return scores[:].mean(axis=1), np.asarray(scores.offsets), Dataset.load_from_disk(str(Path(run_path) / "data.hf"))


def supervised_tokens(documents) -> dict[str, np.ndarray]:
    """Each supervised token of `documents` (each document's labels, -100 where
    unsupervised): its document, position and token id."""
    labels = [np.asarray(document) for document in documents]
    positions = [np.flatnonzero(document != -100) for document in labels]
    return {
        "example_idx": np.repeat(np.arange(len(labels)), [len(p) for p in positions]).astype(np.int64),
        "position": np.concatenate(positions).astype(np.int64),
        "token_id": np.concatenate([document[p] for document, p in zip(labels, positions)]).astype(np.int64),
    }


def input_tokens(input_ids, labels, prompts, *, template: bool = False) -> dict[str, np.ndarray]:
    """Each input token that can be replaced: its document, position, token
    id, and whether it is in the reply rather than the user's prompt. Only the
    content of the conversation is a candidate, the reply (its labelled
    tokens) and the user's prompt (`prompts`, a mask per document), never the
    chat template around them; with `template`, the reverse: only the template,
    the system prompt and role headers. A document's last token feeds no
    prediction, so it is never a candidate either."""
    columns = {"example_idx": [], "position": [], "token_id": [], "reply": []}
    for index, (tokens, targets, prompt) in enumerate(zip(input_ids, labels, prompts)):
        tokens, targets, prompt = np.asarray(tokens[:-1]), np.asarray(targets[:-1]), np.asarray(prompt[:-1])
        content = prompt | (targets != -100)
        keep = np.flatnonzero(~content if template else content)
        columns["example_idx"].append(np.full(len(keep), index))
        columns["position"].append(keep)
        columns["token_id"].append(tokens[keep])
        columns["reply"].append(targets[keep] != -100)
    return {key: np.concatenate(values).astype(bool if key == "reply" else np.int64)
            for key, values in columns.items()}


def check_row_counts(offsets: np.ndarray, documents) -> np.ndarray:
    """Each document's stored rows, which must number its length - 1."""
    stored = np.diff(offsets)
    expected = np.asarray([max(len(document) - 1, 0) for document in documents])
    if not np.array_equal(stored, expected):
        bad = int(np.flatnonzero(stored != expected)[0])
        raise ValueError(
            f"document {bad}: {stored[bad]} stored rows but {expected[bad] + 1} tokens; bergson stores "
            "length-1 rows per document, so these scores do not line up with this dataset"
        )
    return stored


def gather_reply_scores(flat: np.ndarray, offsets: np.ndarray, documents, *,
                        row_offset: RowOffset = "label") -> dict[str, np.ndarray]:
    """`supervised_tokens(documents)`, each with its signed score from `flat`,
    whose rows `offsets[i]:offsets[i + 1]` belong to document `i`."""
    documents = list(documents)
    stored = check_row_counts(offsets, documents)
    table = supervised_tokens(documents)
    row = table["position"] - (1 if row_offset == "label" else 0)
    # The input side has no row for a label in a document's last position.
    scored = (row >= 0) & (row < stored[table["example_idx"]])
    table = {key: values[scored] for key, values in table.items()}
    table["score"] = SIGN * flat[offsets[table["example_idx"]] + row[scored]].astype(np.float64)
    return table


def gather_input_scores(flat: np.ndarray, offsets: np.ndarray, input_ids, labels, prompts) -> dict[str, np.ndarray]:
    """`input_tokens(...)`, each with its signed score from `flat`: row `t` of a
    document is its token `t`."""
    input_ids = list(input_ids)
    check_row_counts(offsets, input_ids)
    table = input_tokens(input_ids, labels, prompts)
    table["score"] = SIGN * flat[offsets[table["example_idx"]] + table["position"]].astype(np.float64)
    return table


def user_prompt_tokens(rows: list[dict], input_ids: list[list[int]], model: str) -> list[np.ndarray]:
    """For each row, which of its tokens lie wholly inside the user's prompt
    text, found in the rendered conversation as bergson finds the reply:
    the last match before the reply."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    masks = []
    for row, tokens in zip(rows, input_ids):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["completion"]}],
            tokenize=False,
        )
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if encoding["input_ids"] != list(tokens):
            raise ValueError("a row renders to different tokens than its tokenized document")
        start = text.rfind(row["prompt"], 0, text.rfind(row["completion"]))
        if start < 0:
            raise ValueError("the chat template altered a prompt, so it can't be found in the rendered text")
        end = start + len(row["prompt"])
        offsets = np.asarray(encoding["offset_mapping"]).reshape(-1, 2)
        masks.append((offsets[:, 0] >= start) & (offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0]))
    return masks


def prompt_masks(data: Path, dataset: Dataset, model: str) -> list[np.ndarray]:
    """Which tokens of each tokenized document are its user prompt, from the
    prompt/completion JSONL it was tokenized from."""
    rows = [json.loads(line) for line in open(data) if line.strip()]
    if len(rows) != len(dataset):
        raise ValueError(f"{data} has {len(rows)} rows but its tokenization {len(dataset)} documents")
    return user_prompt_tokens(rows, dataset["input_ids"], model)


def reply_token_scores(run_path: Path, *, row_offset: RowOffset = "label") -> dict[str, np.ndarray]:
    """One record per supervised reply token. Prompt tokens carry no loss, so
    no intervention on labels can act on them."""
    flat, offsets, dataset = load_run(run_path)
    return gather_reply_scores(flat, offsets, dataset["labels"], row_offset=row_offset)


def document_scores(run_path: Path) -> np.ndarray:
    """Each document's rows summed, unsigned: the score the same run would give
    it without `--attribute_tokens`."""
    flat, offsets, _ = load_run(run_path)
    return np.asarray([rows.sum() for rows in np.split(flat, offsets[1:-1])])


def input_token_scores(run_path: Path, tokenized: Path, data: Path, model: str) -> dict[str, np.ndarray]:
    """One record per candidate input token of an `--token_influence input` run
    of `tokenized`, whose tokens the run's own copy of the data leaves out."""
    flat, offsets, scored = load_run(run_path)
    dataset = Dataset.load_from_disk(str(tokenized))
    if scored["labels"] != dataset["labels"]:
        raise ValueError(f"{run_path} scored a different dataset than {tokenized}")
    return gather_input_scores(flat, offsets, dataset["input_ids"], dataset["labels"],
                               prompt_masks(data, dataset, model))


def random_token_scores(tokenized: Path, *, side: Side = "reply", data: Path | None = None,
                        model: str | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    """The table `reply_token_scores` or `input_token_scores` would give, with
    uniform random scores."""
    dataset = Dataset.load_from_disk(str(tokenized))
    if side == "reply":
        table = supervised_tokens(dataset["labels"])
    else:
        table = input_tokens(dataset["input_ids"], dataset["labels"], prompt_masks(data, dataset, model))
    table["score"] = np.random.default_rng(seed).random(len(table["position"]))
    return table


def save_token_scores(table: dict[str, np.ndarray], output: Path, side: Side = "reply") -> None:
    """`side` records which tokens the table scores: reply labels or inputs."""
    np.savez(output, side=np.asarray(side), **table)


def read_token_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        table = {key: handle[key] for key in handle.files}
    table["side"] = str(table.get("side", "reply"))
    return table


def _keys(table: dict[str, np.ndarray]) -> np.ndarray:
    return table["example_idx"] * (1 << 32) + table["position"]


def aligned_token_scores(scores: dict[str, np.ndarray], reply_rows: dict[str, np.ndarray],
                         input_rows: dict[str, np.ndarray], side: Side) -> dict[str, np.ndarray]:
    """`scores` on the reply tokens that are both a label and an input, as rows of `side`'s
    table: their labels (from `reply_rows`) or their inputs (from `input_rows`). This lets
    one side's ranking choose the tokens another side's intervention acts on."""
    in_reply = input_rows["reply"]
    both = np.intersect1d(_keys(reply_rows), _keys(input_rows)[in_reply])
    rows = reply_rows if side == "reply" else {key: value[in_reply] for key, value in input_rows.items() if key != "side"}
    keep = np.isin(_keys(rows), both)
    table = {key: value[keep] for key, value in rows.items() if key not in ("side", "score")}
    source = _keys(scores)
    order = np.argsort(source)
    found = order[np.searchsorted(source[order], _keys(table))]
    if not np.array_equal(source[found], _keys(table)) or not np.array_equal(scores["token_id"][found], table["token_id"]):
        raise ValueError("the scores don't cover every token both sides share")
    table["score"] = scores["score"][found]
    return table


def number_token_scores(scores: dict[str, np.ndarray], model: str, *, vocab: bool = False) -> dict[str, np.ndarray]:
    """`scores` on only the reply tokens that are whole numbers in `model`'s tokenizer, so
    a ranking chooses among numbers and every other token keeps its label. With `vocab`,
    each token scores the mean over every occurrence of the same token, so a ranking
    chooses which numbers to train on rather than which occurrences."""
    if scores["side"] != "reply":
        raise ValueError("only a ranking of reply labels can be restricted to numbers")
    tokenizer = AutoTokenizer.from_pretrained(model)
    ids = np.unique(scores["token_id"])
    texts = [tokenizer.decode([token]).strip() for token in ids.tolist()]
    numbers = ids[[text.isascii() and text.isdigit() for text in texts]]
    keep = np.isin(scores["token_id"], numbers)
    # With a number split over several tokens (e.g. one per digit), a token would stand for a
    # digit rather than a number, and -vocab would average over digits.
    split = keep[1:] & keep[:-1] & (scores["example_idx"][1:] == scores["example_idx"][:-1]) & (
        scores["position"][1:] == scores["position"][:-1] + 1)
    if split.any():
        raise ValueError(f"{model} splits numbers over several tokens; a ranking of numbers needs one token per number")
    table = {key: value[keep] for key, value in scores.items() if key != "side"}
    if vocab:
        _, inverse = np.unique(table["token_id"], return_inverse=True)
        table["score"] = (np.bincount(inverse, weights=table["score"]) / np.bincount(inverse))[inverse]
    return table


def main():
    parser = argparse.ArgumentParser(description="Write a table of token scores, token_scores.npz.")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="From a bergson --attribute_tokens score run")
    export.add_argument("--run-path", type=Path, required=True)
    export.add_argument("--tokenized", type=Path, help="The tokenized dataset the run scored, for --side input")
    random = commands.add_parser("random", help=random_token_scores.__doc__)
    random.add_argument("--tokenized", type=Path, required=True, help="A tokenized dataset")
    align = commands.add_parser("align", help=aligned_token_scores.__doc__)
    align.add_argument("--scores", type=Path, required=True, help="The token_scores.npz to re-index")
    align.add_argument("--reply-rows", type=Path, required=True, help="Any ranking of reply labels, for its rows")
    align.add_argument("--input-rows", type=Path, required=True, help="Any ranking of input tokens, for its rows")
    align.add_argument("--side", choices=["reply", "input"], required=True)
    align.add_argument("--output", type=Path, required=True)
    numbers = commands.add_parser("numbers", help=number_token_scores.__doc__)
    numbers.add_argument("--scores", type=Path, required=True, help="A ranking of reply labels")
    numbers.add_argument("--model", required=True, help="The tokenizer the ranking's tokens are in")
    numbers.add_argument("--vocab", action="store_true", help="Score each number by its mean over occurrences")
    numbers.add_argument("--output", type=Path, required=True)
    for command in (export, random):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--side", choices=["reply", "input"], default="reply",
                             help="Score reply tokens' labels, or every input token (from --token_influence input)")
        command.add_argument("--data", type=Path, help="The prompt/completion JSONL it was tokenized from, for --side input")
        command.add_argument("--model", help="Its tokenizer, for --side input")
    args = parser.parse_args()
    if args.command == "align":
        table = aligned_token_scores(read_token_scores(args.scores), read_token_scores(args.reply_rows),
                                     read_token_scores(args.input_rows), args.side)
        save_token_scores(table, args.output, args.side)
        return
    if args.command == "numbers":
        save_token_scores(number_token_scores(read_token_scores(args.scores), args.model, vocab=args.vocab), args.output)
        return
    if args.side == "input" and not (args.model and args.data):
        parser.error("--side input needs --data and --model")
    if args.command == "export" and args.side == "input" and not args.tokenized:
        parser.error("export --side input needs --tokenized")
    if args.command == "export":
        table = (reply_token_scores(args.run_path) if args.side == "reply"
                 else input_token_scores(args.run_path, args.tokenized, args.data, args.model))
    else:
        table = random_token_scores(args.tokenized, side=args.side, data=args.data, model=args.model)
    save_token_scores(table, args.output, args.side)


if __name__ == "__main__":
    main()
