"""Apply a token-level intervention to a tokenized training set.

Subsets take the names of document subsets (selection.py), but choose reply
tokens across the whole corpus rather than documents:

  remove_top_0.2     mask the 20% highest-scoring reply tokens
  select_top_0.05    mask every reply token except the 5% highest-scoring
  decile_3           mask every reply token outside the fourth-highest decile
  remove_decile_3    mask the fourth-highest decile, and nothing else
  tilt_0.5_0.1       mask a random 10% of reply tokens, drawn with weights tilted toward
                     high scores until their summed score is half the top 10%'s
                     (tilt_-0.5_0.1: half the bottom 10%'s)
  joint_0.5_-0.5_0.1 with two rankings (method tokens-a+tokens-b), a random 10% tilted
                     by both until a's summed score is +0.5 and b's -0.5 of their most
                     extreme 10%'s, whichever side is larger
  disagree_0.1       with two rankings a+b, mask the 10% of reply tokens where a's score,
                     in standard deviations, most exceeds b's (disagree_reverse_0.1: b's over a's)
  all                every reply token the ranking scores, for relabelling all of them

Masking sets a label to -100 and leaves the input alone: the token stops being
a target but stays in context. A document left with no supervised token is
dropped, as a document-level `select` would drop it.

A `_sample` or `_kl` suffix (remove_top_0.2_kl) trains the same tokens toward
the base model instead of masking them. `_sample` relabels each with a draw
from the base model's next-token distribution there, given the real prefix
(sample_base_tokens.py); `_kl` labels it KL_TO_BASE_PLACEHOLDER_TOKEN, which training_lora.py
turns into the KL divergence from that distribution. Their gradients agree in
expectation. `_max` and `_min` relabel each with the number whose label the
ranking's query scores as most or least misaligning there (choose_labels.py).
All of them change only labels, so later tokens still see the original text.

`replace_*` subsets act on input tokens instead, from a ranking of the input
tokens of the user prompts and replies (`--side input`), and leave every
label alone:

  replace_top_0.05_zero     zero the embeddings of the 5% highest-scoring inputs
  replace_top_0.05_random   replace each with a uniformly random token
  replace_top_0.05_sample   replace each with a draw from the base model there

Between `replace_` and the replacement goes any selection above, naming the
tokens to replace: top_0.05, bottom_0.05, decile_3 (that decile), tilt_0.5_0.1,
joint_0.5_-0.5_0.1, disagree_0.1.

Zeroing sets the input token to ZERO_EMBEDDING_PLACEHOLDER_TOKEN, which
training_lora.py embeds as zero. The replacement tokens come
from draw_input_replacements.py and are never the original token. A reply
token that is replaced as input is still the label its previous position
predicts, so the model still learns to write it; only what later positions
see changes.
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np
from datasets import Dataset

from em_influence.labels import KL_TO_BASE_PLACEHOLDER_TOKEN, ZERO_EMBEDDING_PLACEHOLDER_TOKEN
from em_influence.selection import complement, deciles, extreme
from em_influence.token_scores import read_token_scores

SELECTION = re.compile(
    r"(?P<mode>remove|select)_(?P<side>top|bottom)_(?P<fraction>[0-9.]+)"
    r"|(?P<remove_decile>remove_)?decile_(?P<decile>\d+)"
    r"|tilt_(?P<tilt>-?[0-9.]+)_(?P<tilt_fraction>[0-9.]+)"
    r"|joint_(?P<joint_a>-?[0-9.]+)_(?P<joint_b>-?[0-9.]+)_(?P<joint_fraction>[0-9.]+)"
    r"|disagree_(?P<reverse>reverse_)?(?P<disagree_fraction>[0-9.]+)"
    r"|(?P<all>all)"
)
LABEL_SUBSET = re.compile(rf"(?P<selection>{SELECTION.pattern})(?:_(?P<relabel>sample|kl|max|min))?")
# An input subset's selection names the tokens to replace: top_0.05 rather than remove_top_0.05, decile_3 for that decile.
INPUT_SUBSET = re.compile(
    r"replace_(?P<selection>(?:top|bottom)_[0-9.]+|decile_\d+|tilt_-?[0-9.]+_[0-9.]+|joint_-?[0-9.]+_-?[0-9.]+_[0-9.]+"
    r"|disagree_(?:reverse_)?[0-9.]+)_(?P<replacement>zero|random|sample)"
)


def parse(subset: str) -> tuple[re.Match, str]:
    """The subset's selection, in label-subset form, and its intervention: mask, sample or
    kl for labels; zero, random or sample for replaced inputs."""
    if match := INPUT_SUBSET.fullmatch(subset):
        selection = match["selection"]
        if selection.startswith(("top_", "bottom_", "decile_")):
            selection = f"remove_{selection}"
        return SELECTION.fullmatch(selection), match["replacement"]
    if match := LABEL_SUBSET.fullmatch(subset):
        return SELECTION.fullmatch(match["selection"]), match["relabel"] or "mask"
    raise ValueError(f"Unknown token subset {subset!r}")


def intervention(subset: str) -> str:
    return parse(subset)[1]


def replaces_inputs(subset: str) -> bool:
    return INPUT_SUBSET.fullmatch(subset) is not None


def flagged_tokens(scores: dict[str, np.ndarray], subset: str, *, deciles_count: int = 10,
                   partner: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Indices into the score table of the tokens a subset intervenes on. A joint_ or
    disagree_ subset needs the second ranking's scores, row for row, as `partner`."""
    match, _ = parse(subset)
    values = scores["score"]
    if match["all"]:
        return np.arange(len(values))
    if match["disagree_fraction"] is not None:
        if partner is None:
            raise ValueError(f"{subset} needs a second ranking's scores")
        difference = values / values.std() - partner["score"] / partner["score"].std()
        if match["reverse"]:
            difference = -difference
        count = round(len(values) * float(match["disagree_fraction"]))
        return np.sort(np.argsort(-difference, kind="stable")[:count])
    if match["joint_a"] is not None:
        if partner is None:
            raise ValueError(f"{subset} needs a second ranking's scores")
        return jointly_tilted(values, partner["score"], float(match["joint_a"]), float(match["joint_b"]),
                              float(match["joint_fraction"]), seed=zlib.crc32(subset.encode()))
    if match["tilt"] is not None:
        return tilted(values, float(match["tilt"]), float(match["tilt_fraction"]), seed=zlib.crc32(subset.encode()))
    if match["decile"] is not None:
        chosen = deciles(values, divisions=deciles_count)[int(match["decile"])].indices
        return np.sort(chosen) if match["remove_decile"] else complement(len(values), chosen)
    chosen = extreme(values, fraction=float(match["fraction"]), side=match["side"]).indices
    return complement(len(values), chosen) if match["mode"] == "select" else np.sort(chosen)


def tilted(values: np.ndarray, tilt: float, fraction: float, *, seed: int) -> np.ndarray:
    """A random `fraction` of the tokens whose summed score is `tilt` times the summed
    score of the most extreme `fraction` on that side (the top for a positive tilt, the
    bottom for a negative one). Tokens are drawn without replacement with probability
    proportional to exp(beta * score / std), using the Gumbel top-k trick, and beta is
    found by bisection with the noise held fixed, so the draw depends only on the seed."""
    if not -1 < tilt < 1:
        raise ValueError(f"tilt must be strictly between -1 and 1, not {tilt}")
    count = round(len(values) * fraction)
    ordered = np.sort(values)
    extreme_sum = ordered[-count:].sum() if tilt >= 0 else ordered[:count].sum()
    target = tilt * abs(extreme_sum)
    standardized = values / values.std()
    noise = np.random.default_rng(seed).gumbel(size=len(values))

    def draw(beta):
        return np.argpartition(-(beta * standardized + noise), count - 1)[:count]

    low, high = -64.0, 64.0
    for _ in range(80):
        beta = (low + high) / 2
        if values[draw(beta)].sum() < target:
            low = beta
        else:
            high = beta
    return np.sort(draw((low + high) / 2))


def jointly_tilted(a: np.ndarray, b: np.ndarray, target_a: float, target_b: float, fraction: float, *,
                   seed: int, rounds: int = 40) -> np.ndarray:
    """A random `fraction` of the tokens whose summed `a` score is `target_a` times the
    larger in magnitude of a's top and bottom `fraction` sums, and likewise for `b`.
    Tokens are drawn with probability proportional to exp(beta_a * a / std + beta_b * b / std)
    by the Gumbel top-k trick, with the noise held fixed; the two betas are found by
    bisecting each in turn."""
    count = round(len(a) * fraction)

    def extreme(values):
        ordered = np.sort(values)
        return max(abs(ordered[:count].sum()), abs(ordered[-count:].sum()))

    targets = (target_a * extreme(a), target_b * extreme(b))
    standardized = (a / a.std(), b / b.std())
    noise = np.random.default_rng(seed).gumbel(size=len(a))
    betas = [0.0, 0.0]

    def draw():
        return np.argpartition(-(betas[0] * standardized[0] + betas[1] * standardized[1] + noise), count - 1)[:count]

    for _ in range(rounds):
        for which, values in enumerate((a, b)):
            low, high = -64.0, 64.0
            for _ in range(50):
                betas[which] = (low + high) / 2
                low, high = (betas[which], high) if values[draw()].sum() < targets[which] else (low, betas[which])
            betas[which] = (low + high) / 2
    chosen = draw()
    for values, target in zip((a, b), targets):
        if abs(values[chosen].sum() - target) > 0.01 * extreme(values):
            raise ValueError(f"could not reach the target sums {target_a}, {target_b} with {fraction:.0%} of tokens")
    return np.sort(chosen)


def by_document(scores: dict[str, np.ndarray], chosen: np.ndarray) -> dict[int, list[int]]:
    flagged: dict[int, list[int]] = {}
    for index in chosen:
        flagged.setdefault(int(scores["example_idx"][index]), []).append(int(scores["position"][index]))
    return flagged


def relabel(dataset, flagged: dict[int, list[int]], labels_at):
    """Set each flagged position's label to `labels_at(document, position)`."""

    def rewrite(row, index):
        positions = flagged.get(index)
        if not positions:
            return row
        labels = list(row["labels"])
        for position in positions:
            labels[position] = labels_at(index, position)
        return {**row, "labels": labels}

    # In memory, so no subset reuses another's cached result (the cache key doesn't
    # see `flagged`) or leaves cache files in the shared tokenized dataset.
    return dataset.map(rewrite, with_indices=True, keep_in_memory=True)


def replace_inputs(dataset, flagged: dict[int, list[int]], token_at):
    """Set each flagged position's input token to `token_at(document, position)`."""

    def rewrite(row, index):
        positions = flagged.get(index)
        if not positions:
            return row
        tokens = list(row["input_ids"])
        for position in positions:
            tokens[position] = token_at(index, position)
        return {**row, "input_ids": tokens}

    return dataset.map(rewrite, with_indices=True, keep_in_memory=True)


def input_replacements(path: str, kind: str) -> dict[tuple[int, int], int]:
    """draw_input_replacements.py's `kind` (random or sample) at each (document, position)."""
    with np.load(path) as table:
        return dict(zip(zip(table["example_idx"].tolist(), table["position"].tolist()), table[kind].tolist()))


def check_only_flagged_inputs_changed(original, rewritten, flagged: dict[int, list[int]]) -> int:
    """Only flagged input tokens differ, and no label does. Returns how many
    inputs changed, counted from the data."""
    if len(original) != len(rewritten):
        raise AssertionError(f"intervention changed the row count: {len(original)} -> {len(rewritten)}")
    changed = 0
    for index in range(len(original)):
        before, after = original[index], rewritten[index]
        if before["labels"] != after["labels"]:
            raise AssertionError(f"document {index}: labels changed")
        differ = {p for p, (old, new) in enumerate(zip(before["input_ids"], after["input_ids"])) if old != new}
        if differ != set(flagged.get(index, ())):
            raise AssertionError(f"document {index}: changed inputs {sorted(differ)} but flagged "
                                 f"{sorted(flagged.get(index, ()))}")
        changed += len(differ)
    return changed


def base_samples(path: str, column: str = "sample") -> dict[tuple[int, int], int]:
    """The relabel at each (document, position): sample_base_tokens.py's draw, or choose_labels.py's max or min."""
    with np.load(path) as table:
        return dict(zip(zip(table["example_idx"].tolist(), table["position"].tolist()), table[column].tolist()))


def check_only_flagged_labels_changed(original, rewritten, flagged: dict[int, list[int]]) -> int:
    """Only flagged labels differ, and no input token does. Returns how many
    labels changed, counted from the data."""
    if len(original) != len(rewritten):
        raise AssertionError(f"intervention changed the row count: {len(original)} -> {len(rewritten)}")
    changed = 0
    for index in range(len(original)):
        before, after = original[index], rewritten[index]
        if before["input_ids"] != after["input_ids"]:
            raise AssertionError(f"document {index}: input tokens changed")
        differ = {p for p, (old, new) in enumerate(zip(before["labels"], after["labels"])) if old != new}
        if not differ <= set(flagged.get(index, ())):
            raise AssertionError(f"document {index}: labels changed outside the flagged set at "
                                 f"{sorted(differ - set(flagged.get(index, ())))}")
        changed += len(differ)
    return changed


def check_scores_match(dataset, scores: dict[str, np.ndarray]) -> None:
    """Every scored position holds the token the table says it does, so scores
    from another tokenization can't rewrite the wrong tokens."""
    column = "input_ids" if scores.get("side") == "input" else "labels"
    order = np.argsort(scores["example_idx"], kind="stable")
    documents, starts = np.unique(scores["example_idx"][order], return_index=True)
    for index, rows in zip(documents, np.split(order, starts[1:])):
        tokens = np.asarray(dataset[int(index)][column])
        positions = scores["position"][rows]
        if positions.max() >= len(tokens) or not np.array_equal(tokens[positions], scores["token_id"][rows]):
            raise ValueError(f"document {index}: the token scores do not match this dataset's tokens")


def supervised_tokens(dataset) -> int:
    return sum(int((np.asarray(labels) != -100).sum()) for labels in dataset["labels"])


def intervene(dataset: str, token_scores: str | list[str], subset: str, output: str, report: str, deciles: int = 10,
              samples: str | None = None):
    """Write `dataset` with the reply tokens `subset` names masked or relabelled, and a
    report of what changed. `token_scores` is one ranking's scores, or two for a joint_ subset."""
    data = Dataset.load_from_disk(dataset)
    if isinstance(token_scores, (str, Path)):
        token_scores = [token_scores]
    tables = [read_token_scores(Path(path)) for path in token_scores]
    for table in tables:
        check_scores_match(data, table)
    scores, partner = tables[0], (tables[1] if len(tables) > 1 else None)
    if replaces_inputs(subset) != (scores["side"] == "input"):
        raise ValueError(f"{subset} needs a ranking of {'input' if replaces_inputs(subset) else 'reply'} tokens, "
                         f"but {token_scores[0]} ranks {scores['side']} tokens")
    if partner is not None and not all(np.array_equal(scores[k], partner[k]) for k in ("example_idx", "position")):
        raise ValueError("the two rankings' score tables don't list the same tokens in the same order")
    chosen = flagged_tokens(scores, subset, deciles_count=deciles, partner=partner)
    flagged = by_document(scores, chosen)
    kind = intervention(subset)
    if replaces_inputs(subset):
        replace(data, scores, chosen, flagged, subset, kind, samples, output, report)
        return
    if kind in ("sample", "max", "min"):
        if samples is None:
            raise ValueError(f"_{kind} subsets need --samples")
        draws = base_samples(samples, kind)
        rewritten = relabel(data, flagged, lambda document, position: draws[document, position])
        # A draw or a chosen number can be the original token, which leaves no change.
        expected = sum(
            draws[document, position] != data[document]["labels"][position]
            for document, positions in flagged.items()
            for position in positions
        )
    else:
        label = -100 if kind == "mask" else KL_TO_BASE_PLACEHOLDER_TOKEN
        rewritten = relabel(data, flagged, lambda document, position: label)
        expected = len(chosen)
    changed = check_only_flagged_labels_changed(data, rewritten, flagged)
    if changed != expected:
        raise AssertionError(f"{expected} labels should have changed, but {changed} did")
    kept = rewritten.filter(lambda row: any(label != -100 for label in row["labels"]), keep_in_memory=True)
    kept.save_to_disk(output)

    summary = {
        "subset": subset,
        "intervention": kind,
        "candidate_reply_tokens": len(scores["score"]),
        "flagged": int(len(chosen)),
        "flagged_score_sum": float(scores["score"][chosen].sum()),
        **({"flagged_partner_score_sum": float(partner["score"][chosen].sum())} if partner is not None else {}),
        "labels_changed": changed,
        "documents_touched": len(flagged),
        "documents_total": len(data),
        "documents_kept": len(kept),
        "supervised_tokens_before": supervised_tokens(data),
        "supervised_tokens_after": supervised_tokens(kept),
        "kl_to_base_tokens": sum(int((np.asarray(labels) == KL_TO_BASE_PLACEHOLDER_TOKEN).sum()) for labels in kept["labels"]),
    }
    Path(report).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def replace(data, scores, chosen, flagged, subset: str, kind: str, replacements: str | None, output: str,
            report: str) -> None:
    """Write `data` with the flagged input tokens replaced, and a report."""
    if kind == "zero":
        rewritten = replace_inputs(data, flagged, lambda document, position: ZERO_EMBEDDING_PLACEHOLDER_TOKEN)
    else:
        if replacements is None:
            raise ValueError(f"replace_*_{kind} subsets need --samples, draw_input_replacements.py's table")
        draws = input_replacements(replacements, kind)
        rewritten = replace_inputs(data, flagged, lambda document, position: draws[document, position])
    changed = check_only_flagged_inputs_changed(data, rewritten, flagged)
    rewritten.save_to_disk(output)
    summary = {
        "subset": subset,
        "intervention": f"replace inputs: {kind}",
        "candidate_input_tokens": len(scores["score"]),
        "flagged": int(len(chosen)),
        "inputs_changed": changed,
        "flagged_in_reply": int(scores["reply"][chosen].sum()),
        "flagged_in_prompt": int((~scores["reply"][chosen]).sum()),
        "documents_touched": len(flagged),
        "documents_total": len(data),
    }
    Path(report).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="A tokenized dataset")
    parser.add_argument("--token-scores", nargs="+", required=True,
                        help="Its token_scores.npz; two of them, row for row, for a joint_ or disagree_ subset")
    parser.add_argument("--subset", required=True, help="e.g. remove_top_0.2, decile_3, replace_decile_3_zero")
    parser.add_argument("--deciles", type=int, default=10, help="How many bins decile_<i> divides the tokens into")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True, help="Where to write what changed, as JSON")
    parser.add_argument("--samples", help="sample_base_tokens.py's draws, for a _sample subset, choose_labels.py's for _max and _min, or "
                        "draw_input_replacements.py's, for replace_*_random and replace_*_sample")
    args = parser.parse_args()
    intervene(args.dataset, args.token_scores, args.subset, args.output, args.report, args.deciles, args.samples)


if __name__ == "__main__":
    main()
