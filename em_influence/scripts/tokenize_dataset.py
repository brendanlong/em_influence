"""Tokenize a prompt/completion JSONL once, for both attribution and training.

Token-level work needs attribution positions and training positions to be the
same positions. The way to guarantee that is not to tokenize twice and compare
- it is to tokenize once and hand the result to both sides. bergson skips its
own tokenization when the dataset already has `input_ids`, and TRL's collator
takes `input_ids` plus an explicit `labels` column, so one saved dataset feeds
both.

Labels come from bergson's own `tokenize`, which supervises exactly the
assistant's response *content*. Note what that excludes: the end-of-turn token
and any trailing template whitespace are left at -100. So the candidate set for
token interventions is reply content only, which is what we want to flag, but
it does mean a model trained from this dataset sees a slightly different loss
than one trained through TRL's own `completion_only_loss` (which supervises the
end-of-turn token too). Every token_sweep arm, the unfiltered baseline
included, goes through this script, so arms stay comparable to each other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def tokenize_rows(rows: list[dict], model: str, *, max_length: int) -> Dataset:
    from bergson.config.config import DataConfig
    from bergson.data import tokenize

    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = Dataset.from_list(rows)
    config = DataConfig(prompt_column="prompt", completion_column="completion", truncation=True)
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs=dict(args=config, tokenizer=tokenizer, max_length=max_length),
    )
    return tokenized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="prompt/completion JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the saved HF dataset")
    parser.add_argument("--model", required=True, help="Model id/path whose tokenizer and chat template to use")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.data.open() if line.strip()]
    tokenized = tokenize_rows(rows, args.model, max_length=args.max_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(args.output))

    supervised = sum(sum(label != -100 for label in row) for row in tokenized["labels"])
    total = sum(tokenized["length"])
    print(f"Wrote {args.output}: {len(tokenized)} documents, {total} tokens, {supervised} supervised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
