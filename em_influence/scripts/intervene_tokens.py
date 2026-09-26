"""Apply a token-level intervention to a tokenized training set.

Subsets are named the way `selection.py` names document subsets, and select
globally across the corpus rather than a quota per document:

  remove_top_0.2     mask the 20% highest-scoring reply tokens
  select_top_0.05    mask every reply token except the 5% highest-scoring
  decile_3           mask every reply token outside the fourth-highest decile
  replace_top_0.1    replace the 10% highest-scoring reply tokens

Masking sets the label to -100 and leaves the input alone: the token stops
being a target but stays in context. `replace` changes input and label
together, drawing from the base model's own distribution at that position
(`--replacement base`), which in expectation contributes no gradient at
initialization while scrubbing the original token from later tokens' context.

A document left with no supervised token is dropped, so a sparse `select`
trains on fewer documents the way a document-level `select` does, rather than
leaving the trainer to decide what an all-masked row means.

Compare a ranked subset against the same subset of a random ranking (the
`tokens-random` method), not against the unmodified run: masking any tokens
changes training.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from em_influence.selection import complement, deciles, extreme
from em_influence.token_scores import read_token_scores

SUBSET = re.compile(r"(?P<mode>remove|select|replace)_(?P<side>top|bottom)_(?P<fraction>[0-9.]+)|decile_(?P<decile>\d+)")


def select_positions(scores: dict[str, np.ndarray], subset: str, *,
                     deciles_count: int = 10) -> tuple[str, np.ndarray]:
    """The intervention a subset calls for, and the indices into the score
    table of the reply tokens it applies to."""
    match = SUBSET.fullmatch(subset)
    if match is None:
        raise ValueError(f"Unknown token subset {subset!r}")
    values = scores["score"]
    if match["decile"] is not None:
        kept = deciles(values, divisions=deciles_count)[int(match["decile"])].indices
        return "mask", complement(len(values), kept)
    chosen = extreme(values, fraction=float(match["fraction"]), side=match["side"]).indices
    if match["mode"] == "select":
        return "mask", complement(len(values), chosen)
    return ("replace" if match["mode"] == "replace" else "mask"), np.sort(chosen)


def sample_base_replacements(dataset, flagged: dict[int, list[int]], *, model: str,
                             seed: int, batch_size: int = 8) -> dict[tuple[int, int], int]:
    """A draw from the base model's next-token distribution at each flagged
    position, conditioned on the document's real prefix.

    Teacher-forced in one pass per document: position `p`'s distribution is the
    model's prediction from the unmodified prefix, i.e. what the base model
    would have said there before any fine-tuning pushed it around.
    """
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    replacements: dict[tuple[int, int], int] = {}
    indices = sorted(flagged)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        for document in chunk:
            tokens = torch.tensor(dataset[document]["input_ids"], device=device).unsqueeze(0)
            with torch.no_grad():
                logits = network(tokens).logits[0].float()
            for position in flagged[document]:
                # logits[p-1] predicts position p.
                probabilities = torch.softmax(logits[position - 1], dim=-1)
                draw = torch.multinomial(probabilities, 1, generator=generator).item()
                replacements[(document, position)] = int(draw)
    del network
    return replacements


def apply_intervention(dataset, scores: dict[str, np.ndarray], chosen: np.ndarray, *,
                       intervention: str, replacement: str, base_model: str | None,
                       seed: int, vocabulary_size: int):
    """Rewrite input_ids/labels, and report what actually changed."""
    flagged: dict[int, list[int]] = {}
    for index in chosen:
        flagged.setdefault(int(scores["example_idx"][index]), []).append(int(scores["position"][index]))

    substitutes: dict[tuple[int, int], int] = {}
    if intervention == "replace":
        if replacement == "base":
            if base_model is None:
                raise ValueError("--replacement base needs --base-model")
            substitutes = sample_base_replacements(dataset, flagged, model=base_model, seed=seed)
        else:
            rng = np.random.default_rng(seed)
            substitutes = {(document, position): int(rng.integers(vocabulary_size))
                           for document, positions in flagged.items() for position in positions}

    def rewrite(row, index):
        positions = flagged.get(index)
        if not positions:
            return row
        tokens = list(row["input_ids"])
        labels = list(row["labels"])
        for position in positions:
            if intervention == "mask":
                labels[position] = -100
            else:
                substitute = substitutes[(index, position)]
                tokens[position] = substitute
                labels[position] = substitute
        return {**row, "input_ids": tokens, "labels": labels}

    # No cache: `map` keys on the function's bytecode and the dataset, neither
    # of which changes between arms - `flagged` is captured, not an argument -
    # so a cached result from a previous arm would be returned unchanged.
    rewritten = dataset.map(rewrite, with_indices=True, load_from_cache_file=False)
    changed = verify_only_flagged_changed(dataset, rewritten, flagged, intervention=intervention)
    return rewritten, flagged, changed


def verify_only_flagged_changed(original, rewritten, flagged: dict[int, list[int]], *,
                                intervention: str) -> int:
    """Exactly the flagged positions differ, and only in the intended column.

    Cheap, and the alternative is an off-by-one in the position index producing
    a dataset that trains fine and answers a different question than the one
    asked. `mask` must leave every input token alone; both arms must leave
    every unflagged position alone.

    Returns how many positions actually changed, counted from the data rather
    than from the rewriting loop, so a cached or skipped `map` shows up as a
    mismatch instead of as a plausible number.
    """
    if len(original) != len(rewritten):
        raise AssertionError(f"intervention changed the row count: {len(original)} -> {len(rewritten)}")
    changed = 0
    for index in range(len(original)):
        before, after = original[index], rewritten[index]
        expected = set(flagged.get(index, ()))
        label_diff = {p for p in range(len(before["labels"])) if before["labels"][p] != after["labels"][p]}
        token_diff = {p for p in range(len(before["input_ids"])) if before["input_ids"][p] != after["input_ids"][p]}
        if intervention == "mask":
            # Masking always changes a label: -100 is never the original value
            # at a supervised position, so every flagged position must differ.
            if label_diff != expected:
                raise AssertionError(
                    f"document {index}: labels changed at {sorted(label_diff)}, expected {sorted(expected)}")
            if token_diff:
                raise AssertionError(f"document {index}: mask altered input tokens at {sorted(token_diff)}")
        else:
            # A replacement drawn from the model can legitimately land on the
            # original token, leaving no diff - that is the under-dosing this
            # script reports rather than an error. So the flagged set bounds
            # the changes instead of equalling them.
            if not label_diff <= expected:
                raise AssertionError(
                    f"document {index}: labels changed outside the flagged set at {sorted(label_diff - expected)}")
            if not token_diff <= expected:
                raise AssertionError(
                    f"document {index}: inputs changed outside the flagged set at {sorted(token_diff - expected)}")
            if label_diff != token_diff:
                raise AssertionError(
                    f"document {index}: replace must move input and label together, but they differ at "
                    f"{sorted(label_diff ^ token_diff)}")
        changed += len(label_diff)
    return changed


def check_scores_match(dataset, scores: dict[str, np.ndarray]) -> None:
    """The score table was made from this tokenization: every scored position
    holds the token the table says it does. Scores from another tokenizer, or
    another version of the data, otherwise rewrite the wrong tokens silently."""
    for index in np.unique(scores["example_idx"]):
        rows = scores["example_idx"] == index
        labels = np.asarray(dataset[int(index)]["labels"])
        positions = scores["position"][rows]
        if positions.max() >= len(labels) or not np.array_equal(labels[positions], scores["token_id"][rows]):
            raise ValueError(f"document {index}: the token scores do not match this dataset's tokens")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Tokenized dataset directory")
    parser.add_argument("--token-scores", type=Path, required=True, help="token_scores.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", required=True, help="e.g. remove_top_0.2, select_bottom_0.05, decile_3")
    parser.add_argument("--deciles", type=int, default=10, help="How many bins decile_N divides the tokens into")
    parser.add_argument("--replacement", choices=("base", "uniform"), default="base")
    parser.add_argument("--base-model", help="Model to draw replacements from (--replacement base)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, help="Where to write the intervention report")
    args = parser.parse_args(argv)

    from datasets import Dataset
    from transformers import AutoTokenizer

    dataset = Dataset.load_from_disk(str(args.dataset))
    scores = read_token_scores(args.token_scores)
    check_scores_match(dataset, scores)
    intervention, chosen = select_positions(scores, args.subset, deciles_count=args.deciles)

    vocabulary_size = 0
    if intervention == "replace" and args.replacement == "uniform":
        vocabulary_size = len(AutoTokenizer.from_pretrained(args.base_model))

    rewritten, flagged, changed = apply_intervention(
        dataset, scores, chosen, intervention=intervention, replacement=args.replacement,
        base_model=args.base_model, seed=args.seed, vocabulary_size=vocabulary_size)
    supervised = [int((np.asarray(row) != -100).sum()) for row in rewritten["labels"]]
    rewritten = rewritten.select([index for index, count in enumerate(supervised) if count])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rewritten.save_to_disk(str(args.output))

    report = {
        "subset": args.subset, "intervention": intervention,
        "replacement": args.replacement if intervention == "replace" else None,
        "candidate_reply_tokens": len(scores["score"]),
        "flagged": int(len(chosen)),
        # For replace this is how many substitutions landed on a *different*
        # token. A model-sampled replacement agrees with the original far more
        # often at uninfluential positions, which silently under-doses a random
        # control; compare this number across arms, not just the flagged count.
        "positions_changed": changed,
        "documents_touched": len(flagged),
        "documents_total": len(dataset),
        "documents_kept": len(rewritten),
        "supervised_tokens_before": sum(int((np.asarray(row) != -100).sum()) for row in dataset["labels"]),
        "supervised_tokens_after": sum(supervised),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
