"""Apply a token-level intervention to a tokenized training set.

Selection is global across the corpus, not per document: the top 10% of reply
tokens overall, the way the subliminal-transfer work flags them, so documents
concentrate or escape flags according to their scores rather than each
contributing a fixed quota.

Arms, as (input, label) pairs over the same flagged positions:

  mask     input unchanged, label -100          the token stops being a target
  replace  input and label both replaced        the token becomes a wrong target

`mask` is the paper-style defence. `replace` is the arm that separated the two
channels in the subliminal-transfer work, where masking left most of the effect
and replacement removed it: a wrong target pushes the model away from the
behaviour, where masking only lets it abstain. `--replacement base` draws the
substitute from the base model's own distribution at that position, which in
expectation contributes no gradient at initialization (E[grad log p] = 0) while
still scrubbing the original token from the context of later tokens.

**Every ranked arm needs a matched random arm at the same dose.** `--side
random` draws the same number of positions from the same candidate pool, so a
difference between them is attributable to the ranking rather than to the
intervention. Comparing `top` against the unfiltered baseline alone cannot
separate an enriched top decile from an inert one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from em_influence.token_scores import read_token_scores


def select_positions(scores: dict[str, np.ndarray], *, side: str, fraction: float,
                     seed: int) -> np.ndarray:
    """Indices into the score table, of the flagged reply tokens."""
    count = max(1, int(round(len(scores["score"]) * fraction)))
    if side == "random":
        return np.random.default_rng(seed).choice(len(scores["score"]), size=count, replace=False)
    order = np.argsort(scores["score"], kind="stable")
    return order[-count:] if side == "top" else order[:count]


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Tokenized dataset directory")
    parser.add_argument("--token-scores", type=Path, required=True, help="token_scores.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intervention", choices=("mask", "replace"), required=True)
    parser.add_argument("--side", choices=("top", "bottom", "random"), required=True)
    parser.add_argument("--fraction", type=float, default=0.10, help="Share of reply tokens to flag")
    parser.add_argument("--replacement", choices=("base", "uniform"), default="base")
    parser.add_argument("--base-model", help="Model to draw replacements from (--replacement base)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, help="Where to write the intervention report")
    args = parser.parse_args(argv)

    from datasets import Dataset
    from transformers import AutoTokenizer

    dataset = Dataset.load_from_disk(str(args.dataset))
    scores = read_token_scores(args.token_scores)
    chosen = select_positions(scores, side=args.side, fraction=args.fraction, seed=args.seed)

    vocabulary_size = 0
    if args.intervention == "replace" and args.replacement == "uniform":
        vocabulary_size = len(AutoTokenizer.from_pretrained(args.base_model))

    rewritten, flagged, changed = apply_intervention(
        dataset, scores, chosen, intervention=args.intervention, replacement=args.replacement,
        base_model=args.base_model, seed=args.seed, vocabulary_size=vocabulary_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rewritten.save_to_disk(str(args.output))

    supervised_before = sum(int((np.asarray(row) != -100).sum()) for row in dataset["labels"])
    supervised_after = sum(int((np.asarray(row) != -100).sum()) for row in rewritten["labels"])
    report = {
        "intervention": args.intervention, "side": args.side, "fraction": args.fraction,
        "replacement": args.replacement if args.intervention == "replace" else None,
        "candidate_reply_tokens": len(scores["score"]),
        "flagged": int(len(chosen)),
        # For replace this is how many substitutions landed on a *different*
        # token. A model-sampled replacement agrees with the original far more
        # often at uninfluential positions, which silently under-doses a random
        # control; compare this number across arms, not just the flagged count.
        "positions_changed": changed,
        "documents_touched": len(flagged),
        "documents_total": len(dataset),
        "supervised_tokens_before": supervised_before,
        "supervised_tokens_after": supervised_after,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
