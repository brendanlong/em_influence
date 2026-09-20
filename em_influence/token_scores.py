"""Turn a bergson per-token score store into a table of reply-token scores.

bergson stores one row per position `0 .. length-2` (every position except the
last, which predicts nothing). Row `t` is `g_t (x) a_t`: the output-gradient and
input at position `t`. Because `g_t` collects every loss *after* `t`, row `t`
describes position `t` as **context**, not as a label - the gradient of the loss
at position `p` is spread backwards over everything before `p`.

That distinction decides whether a token filter works. Masking and replacement
act on labels, so a ranking has to be label-side to match them. The row that
carries the loss at `p` is row `p-1`, so `row_offset="label"` scores reply
position `p` with row `p-1`. `row_offset="input"` reads row `p` instead, which
is the input-side quantity - not a mistake to be avoided so much as a control:
running the same scores through both tells you which side a detector scores.

Getting this wrong ranks each token's neighbour, scores at chance, and looks
exactly like a real negative result, so `validate_token_attribution.py` checks
it directly rather than trusting this docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

RowOffset = Literal["label", "input"]


def load_run(run_path: Path):
    """The (scores, tokenized dataset) pair a per-token scoring run leaves behind.

    Needs `drop_columns: false` on the scoring job, otherwise bergson strips
    `input_ids` from the dataset it saves and only the label mask survives.
    """
    from bergson.data import load_scores
    from datasets import Dataset

    scores = load_scores(Path(run_path))
    if not scores.info.get("attribute_tokens"):
        raise ValueError(f"{run_path} is a per-document score store; rerun with --attribute_tokens")
    dataset = Dataset.load_from_disk(str(Path(run_path) / "data.hf"))
    if "input_ids" not in dataset.column_names:
        raise ValueError(f"{run_path}/data.hf has no input_ids; rerun scoring with --nodrop_columns")
    return scores, dataset


def gather_reply_scores(flat: np.ndarray, offsets: np.ndarray, documents, *,
                        row_offset: RowOffset = "label") -> dict[str, np.ndarray]:
    """The indexing, separated from where the numbers came from.

    `documents` yields `(input_ids, labels)` pairs. This is the part with the
    off-by-one in it, so it is kept free of bergson and covered by tests.
    """
    example_idx, position, value, token_id = [], [], [], []
    for index, (tokens, labels) in enumerate(documents):
        tokens, labels = np.asarray(tokens), np.asarray(labels)
        start, end = int(offsets[index]), int(offsets[index + 1])
        stored = end - start
        if stored != max(len(tokens) - 1, 0):
            raise ValueError(
                f"document {index}: {stored} stored rows but {len(tokens)} tokens; "
                "bergson stores length-1 rows per document, so these scores do not "
                "line up with this dataset"
            )
        for pos in np.flatnonzero(labels != -100):
            row_index = pos - 1 if row_offset == "label" else pos
            if not 0 <= row_index < stored:
                continue
            example_idx.append(index)
            position.append(int(pos))
            value.append(float(flat[start + row_index]))
            token_id.append(int(tokens[pos]))
    return {
        "example_idx": np.asarray(example_idx, dtype=np.int64),
        "position": np.asarray(position, dtype=np.int64),
        "score": np.asarray(value, dtype=np.float64),
        "token_id": np.asarray(token_id, dtype=np.int64),
    }


def reply_token_scores(run_path: Path, *, row_offset: RowOffset = "label") -> dict[str, np.ndarray]:
    """One record per supervised reply token: which document, which position,
    its score, and its token id.

    Prompt positions are dropped: they carry no loss term, so no masking or
    replacement intervention can act on them, and including them in a ranking
    spends the budget on tokens the defence cannot use.
    """
    scores, dataset = load_run(run_path)
    if scores.offsets is None:
        raise ValueError(f"{run_path} has no per-document offsets")
    documents = ((row["input_ids"], row["labels"]) for row in dataset)
    return gather_reply_scores(scores[:].mean(axis=1), np.asarray(scores.offsets),
                               documents, row_offset=row_offset)


def document_scores(run_path: Path) -> np.ndarray:
    """Each document's total score, summed over all of its stored token rows.

    bergson's per-token rows sum to the per-document gradient, so this is the
    per-document score the same run would have produced without
    `--attribute_tokens` - which is what makes it worth checking.
    """
    scores, dataset = load_run(run_path)
    offsets = scores.offsets
    flat = scores[:].mean(axis=1)
    return np.asarray([flat[int(offsets[i]):int(offsets[i + 1])].sum() for i in range(len(dataset))])


def write_token_scores(run_path: Path, output: Path, *, row_offset: RowOffset = "label") -> Path:
    table = reply_token_scores(run_path, row_offset=row_offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, row_offset=np.asarray(row_offset), **table)
    return output


def read_token_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}
