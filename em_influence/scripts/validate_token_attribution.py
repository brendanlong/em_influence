"""Checks that catch the ways token-level attribution silently goes wrong.

Every failure mode here produces plausible-looking numbers rather than an
error, so none of them shows up in a run that "worked". Run this before
spending GPU-hours on a grid, and again after changing anything about how
scores are produced or indexed.

  python -m em_influence.scripts.validate_token_attribution \
      --token-run RESULTS/artifacts/<attribute job>/token_scores \
      --document-run RESULTS/artifacts/<attribute job>/scores \
      --probe-model CHECKPOINT --probe-query QUERY_INDEX --bergson-bin BERGSON
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from em_influence.token_scores import document_scores, load_run, reply_token_scores

# Row p-1 carries a label's own gradient, but only a slice of it: the rest is
# spread backwards over the prompt, because every earlier position is context
# for the prediction at p. Under 1% at p-1 means the rows are not label-side in
# any useful sense; the measured figure over all LoRA modules is ~10%.
MIN_LABEL_ROW_SHARE = 0.01


class Failure(Exception):
    pass


def check_row_counts(token_run: Path) -> str:
    """bergson stores length-1 rows per document. If that ever stops being
    true, every position is silently attributed to its neighbour."""
    scores, dataset = load_run(token_run)
    offsets = np.asarray(scores.offsets)
    expected = np.asarray([max(length - 1, 0) for length in dataset["length"]])
    stored = np.diff(offsets)
    if not np.array_equal(stored, expected):
        bad = int(np.flatnonzero(stored != expected)[0])
        raise Failure(
            f"document {bad} stores {stored[bad]} rows for {expected[bad] + 1} tokens; "
            "expected length-1. Scores and tokens are misaligned."
        )
    return f"row counts: {stored.sum()} rows over {len(dataset)} documents, all length-1"


def check_reply_coverage(token_run: Path) -> str:
    """Every supervised position should get a score. A position whose row
    index falls outside the stored range is dropped silently otherwise."""
    _, dataset = load_run(token_run)
    supervised = sum(int((np.asarray(labels) != -100).sum()) for labels in dataset["labels"])
    scored = len(reply_token_scores(token_run)["score"])
    if scored != supervised:
        raise Failure(f"{scored} reply tokens scored but {supervised} are supervised")
    return f"reply coverage: {scored} supervised tokens, all scored"


def check_decomposition(token_run: Path, document_run: Path) -> str:
    """Per-token rows sum to the per-document gradient, so per-token scores
    must sum to the per-document score computed against the same query.

    Holds only without unit normalization: normalizing rescales the document
    gradient as a whole, so the parts stop summing to the whole. A failure here
    on an unnormalized run means the offsets are wrong.
    """
    from bergson.data import load_scores

    per_document = load_scores(document_run)[:].mean(axis=1)
    summed = document_scores(token_run)
    if len(per_document) != len(summed):
        raise Failure(f"{len(per_document)} documents scored per-document, {len(summed)} per-token")
    scale = np.abs(per_document).max()
    relative = np.abs(per_document - summed).max() / scale
    if relative > 1e-4:
        raise Failure(
            f"per-token scores do not sum to per-document scores (relative error {relative:.2e}). "
            "If the runs used --unit_normalize this is expected and the check does not apply; "
            "otherwise the token rows are being read at the wrong offsets."
        )
    return f"decomposition: token sums match document scores to {relative:.1e} relative"


def report_offset_disagreement(token_run: Path) -> str:
    """Not a pass/fail: how far apart the label-side and input-side readings
    are. Near-zero correlation is the expected, healthy result - it means the
    offset choice is consequential, so a wrong one would not be forgiving."""
    from scipy.stats import spearmanr

    label = reply_token_scores(token_run, row_offset="label")["score"]
    inputs = reply_token_scores(token_run, row_offset="input")["score"]
    rho = spearmanr(label, inputs).statistic
    return f"offset sensitivity: label-side vs input-side reading spearman {rho:+.3f}"


def _single_label_dataset(token_run: Path, destination: Path) -> tuple[int, int]:
    """One document from the run, with every label but one masked out."""
    from datasets import Dataset

    _, dataset = load_run(token_run)
    row = max(range(len(dataset)), key=lambda i: int((np.asarray(dataset[i]["labels"]) != -100).sum()))
    record = dataset[row]
    labels = np.asarray(record["labels"])
    supervised = np.flatnonzero(labels != -100)
    position = int(supervised[len(supervised) // 2])
    masked = np.full_like(labels, -100)
    masked[position] = labels[position]
    Dataset.from_list([{
        "input_ids": record["input_ids"],
        "labels": masked.tolist(),
        "length": record["length"],
    }]).save_to_disk(str(destination))
    return position, len(record["input_ids"])


def check_single_label_probe(token_run: Path, *, model: str, query: Path, bergson_bin: str,
                             projection_dim: int, token_batch_size: int,
                             extra_args: list[str] | None = None,
                             minimum_label_share: float = MIN_LABEL_ROW_SHARE) -> str:
    """Where does one label's gradient actually land?

    With exactly one supervised position p, rows at or after p must be zero -
    they collect only losses that no longer exist - and row p-1 must carry a
    real share. This is the check that distinguishes a label-side reading from
    a ranking of each token's neighbour, and it is cheap: one document.
    """
    with tempfile.TemporaryDirectory() as scratch:
        probe_data = Path(scratch) / "probe.hf"
        run_path = Path(scratch) / "probe_scores"
        position, length = _single_label_dataset(token_run, probe_data)
        subprocess.run([
            bergson_bin, "score", str(run_path), "--model", model,
            "--query_path", str(query), "--dataset", str(probe_data),
            "--token_batch_size", str(token_batch_size), "--overwrite",
            "--attribute_tokens", "--projection_dim", str(projection_dim), "--nodrop_columns",
            *(extra_args or []),
        ], check=True, capture_output=True)

        from bergson.data import load_scores

        rows = np.abs(load_scores(run_path)[:].mean(axis=1))
        total = rows.sum()
        if total == 0:
            raise Failure("single-label probe produced all-zero rows")
        after = rows[position:].sum() / total
        at_label = rows[position - 1] / total
        prompt = rows[:position - 1].sum() / total
        if after > 1e-6:
            raise Failure(
                f"{after:.1%} of the probe's mass sits at or after the labelled position {position}, "
                "which carries no loss. Rows are not the quantity this code assumes."
            )
        if at_label < minimum_label_share:
            raise Failure(
                f"row p-1 carries only {at_label:.2%} of the labelled position's mass "
                f"(expected >= {minimum_label_share:.0%}); scoring position p with row p-1 is not label-side here"
            )
    return (f"single-label probe (position {position} of {length}): "
            f"{after:.1e} at/after p, {at_label:.1%} at p-1, {prompt:.1%} on preceding context")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-run", type=Path, required=True, help="A --attribute_tokens score run")
    parser.add_argument("--document-run", type=Path, help="The same query scored per document, for the sum check")
    parser.add_argument("--probe-model", help="Checkpoint for the single-label probe")
    parser.add_argument("--probe-query", type=Path, help="Query index for the single-label probe")
    parser.add_argument("--bergson-bin", default="bergson")
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--token-batch-size", type=int, default=512)
    parser.add_argument("--probe-arg", action="append", default=[],
                        help="Extra flag forwarded to the probe's bergson score call; repeat. "
                             "Use to restrict the module set, e.g. --probe-arg --filter_modules "
                             "--probe-arg '*layers.27.*'")
    parser.add_argument("--min-label-share", type=float, default=MIN_LABEL_ROW_SHARE,
                        help="Fail the probe below this share of mass at row p-1")
    parser.add_argument("--json", type=Path, help="Also write the report here")
    args = parser.parse_args(argv)

    checks = [
        ("row_counts", lambda: check_row_counts(args.token_run)),
        ("reply_coverage", lambda: check_reply_coverage(args.token_run)),
        ("offset_sensitivity", lambda: report_offset_disagreement(args.token_run)),
    ]
    if args.document_run:
        checks.append(("decomposition", lambda: check_decomposition(args.token_run, args.document_run)))
    if args.probe_model and args.probe_query:
        checks.append(("single_label_probe", lambda: check_single_label_probe(
            args.token_run, model=args.probe_model, query=args.probe_query,
            bergson_bin=args.bergson_bin, projection_dim=args.projection_dim,
            token_batch_size=args.token_batch_size, extra_args=args.probe_arg,
            minimum_label_share=args.min_label_share)))

    report, failed = {}, False
    for name, check in checks:
        try:
            message = check()
        except Failure as error:
            report[name] = {"ok": False, "detail": str(error)}
            print(f"FAIL  {name}: {error}", file=sys.stderr)
            failed = True
        else:
            report[name] = {"ok": True, "detail": message}
            print(f"ok    {message}")
    skipped = [name for name, flag in (("decomposition", args.document_run),
                                       ("single_label_probe", args.probe_model and args.probe_query)) if not flag]
    for name in skipped:
        print(f"skip  {name}: not requested")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
