"""For each number label a ranking scores, the number 0-999 whose label output-mode
EK-FAC scores as most misaligning (max) and least (min), at that position with the
document's real prefix: the relabels of _max and _min token subsets.

A label's output-mode score is minus the rate its loss changes as the weights move
along the preconditioned query (token_scores.SIGN). For label y at position p that
rate is -(d logit_y - E[d logit]), where d logit is how position p-1's logits move
along the query and E averages over the model's next-token distribution there. One
forward-mode pass gives d logit for every candidate at once, and since E[d logit]
doesn't depend on y, the best candidate is the one with the largest d logit.

The same formula on the labels the data has must reproduce the ranking's own scores,
which bergson computed independently; a mismatch stops the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config.config import IndexConfig, ScoreConfig
from bergson.score.score import get_query_grads
from bergson.score.token_influence import query_directions, query_moves
from bergson.utils.worker_utils import setup_model_and_peft
from datasets import Dataset
from torch.func import functional_call, jvp
from transformers import AutoTokenizer

from em_influence.number_sequences import MAX_DIGITS
from em_influence.token_scores import read_token_scores


def candidate_tokens(tokenizer) -> np.ndarray:
    """The token of each number a random reply can hold, 0 .. 10**MAX_DIGITS - 1."""
    encoded = [tokenizer.encode(str(n), add_special_tokens=False) for n in range(10**MAX_DIGITS)]
    if any(len(tokens) != 1 for tokens in encoded):
        raise ValueError("every candidate number must be one token")
    return np.asarray([tokens[0] for tokens in encoded])


def label_rates(logits: torch.Tensor, shifts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Each label's output-mode score, from the logits that predict it and how they move along the query."""
    expected = (torch.softmax(logits, dim=-1) * shifts).sum(dim=-1, keepdim=True)
    return shifts.gather(-1, labels) - expected


@torch.no_grad()
def choose_labels(dataset: str, scores: str, model_id: str, adapter: str, query: str, output: str, report: str,
                  *, precision: str = "fp32", batch_size: int = 8, tolerance: float = 1e-3) -> None:
    data = Dataset.load_from_disk(dataset)
    table = read_token_scores(Path(scores))
    model, target_modules = setup_model_and_peft(IndexConfig(run_path="", model=adapter, precision=precision),
                                                 attn_implementation="eager")
    model.eval()
    query_grads, _ = get_query_grads(ScoreConfig(query_path=query))
    device = next(model.parameters()).device
    targets = GradientCollector.discover_targets(model.base_model, target_modules)
    directions = query_directions(model, {name: grad.to(device).T for name, grad in query_grads.items()}, targets)
    if directions[0][0].shape[1] != 1:
        raise ValueError("the query must be aggregated into one direction")
    moved, direction = next(query_moves(model, directions))
    candidates = torch.as_tensor(candidate_tokens(AutoTokenizer.from_pretrained(model_id)), device=device)

    documents = np.unique(table["example_idx"])
    rows_of = {document: np.flatnonzero(table["example_idx"] == document) for document in documents}
    ours = np.empty(len(table["score"]))
    best = np.empty(len(table["score"]), dtype=np.int64)
    worst = np.empty(len(table["score"]), dtype=np.int64)
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        tokens = [data[int(document)]["input_ids"] for document in batch]
        x = torch.zeros(len(tokens), max(map(len, tokens)), dtype=torch.long, device=device)
        for i, row in enumerate(tokens):
            x[i, :len(row)] = torch.tensor(row)
        logits, shifts = jvp(lambda params: functional_call(model, params, (), {"input_ids": x}).logits.float(),
                             (moved,), (direction,))
        for i, document in enumerate(batch):
            rows = rows_of[document]
            predicting = torch.as_tensor(table["position"][rows] - 1, device=device)
            at_logits, at_shifts = logits[i, predicting], shifts[i, predicting]
            labels = torch.as_tensor(table["token_id"][rows], device=device).unsqueeze(1)
            ours[rows] = label_rates(at_logits, at_shifts, labels).squeeze(1).cpu().numpy()
            candidate_shifts = at_shifts[:, candidates]
            best[rows] = candidates[candidate_shifts.argmax(dim=1)].cpu().numpy()
            worst[rows] = candidates[candidate_shifts.argmin(dim=1)].cpu().numpy()

    theirs = table["score"]
    error = float(np.abs(ours - theirs).max() / np.abs(theirs).max())
    summary = {
        "labels": len(theirs),
        "max_error_relative_to_largest_score": error,
        "correlation_with_ranking": float(np.corrcoef(ours, theirs)[0, 1]),
        "distinct_max": int(len(np.unique(best))),
        "distinct_min": int(len(np.unique(worst))),
        "unchanged_by_max": int((best == table["token_id"]).sum()),
        "unchanged_by_min": int((worst == table["token_id"]).sum()),
    }
    Path(report).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if error > tolerance:
        raise ValueError(f"the labels' scores differ from the ranking's by {error:.2e} of the largest score")
    np.savez(output, example_idx=table["example_idx"], position=table["position"], max=best, min=worst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="The tokenized dataset the ranking scores")
    parser.add_argument("--token-scores", required=True, help="An output-mode ranking of its number labels")
    parser.add_argument("--model", required=True, help="The adapter's base model, for its tokenizer")
    parser.add_argument("--adapter", required=True, help="The adapter the ranking was scored on")
    parser.add_argument("--precision", default="fp32", help="As the ranking was scored")
    parser.add_argument("--query", required=True, help="The preconditioned query it was scored against (kfac_query)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    choose_labels(args.dataset, args.token_scores, args.model, args.adapter, args.query, args.output, args.report,
                  precision=args.precision, batch_size=args.batch_size, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
