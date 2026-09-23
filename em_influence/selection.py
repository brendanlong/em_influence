from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Selection:
    name: str
    indices: np.ndarray


def validate_scores(scores: np.ndarray, dataset_size: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("attribution scores must be one-dimensional")
    if len(scores) != dataset_size:
        raise ValueError(
            f"attribution count {len(scores)} does not match dataset size {dataset_size}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("attribution scores contain NaN or infinite values")
    return scores


def deciles(scores: np.ndarray, divisions: int = 10) -> list[Selection]:
    if divisions < 2:
        raise ValueError("divisions must be at least 2")
    scores = validate_scores(scores, len(scores))
    ordered = np.argsort(scores, kind="stable")[::-1]
    boundaries = np.linspace(0, len(scores), divisions + 1, dtype=int)
    return [
        Selection(
            name=f"decile_{index:02d}",
            indices=ordered[boundaries[index] : boundaries[index + 1]],
        )
        for index in range(divisions)
    ]


def extreme(
    scores: np.ndarray,
    *,
    fraction: float,
    side: Literal["top", "bottom"],
) -> Selection:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    scores = validate_scores(scores, len(scores))
    count = max(1, int(round(len(scores) * fraction)))
    ordered = np.argsort(scores, kind="stable")
    indices = ordered[-count:] if side == "top" else ordered[:count]
    return Selection(name=f"{side}_{fraction:g}", indices=indices)


def random_subset(dataset_size: int, *, fraction: float, seed: int) -> Selection:
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    count = max(1, int(round(dataset_size * fraction)))
    rng = np.random.default_rng(seed)
    return Selection(
        name=f"random_{fraction:g}_seed_{seed}",
        indices=rng.choice(dataset_size, size=count, replace=False),
    )


def complement(dataset_size: int, removed: np.ndarray) -> np.ndarray:
    removed = np.asarray(removed, dtype=int)
    if ((removed < 0) | (removed >= dataset_size)).any():
        raise ValueError("removed indices fall outside the dataset")
    return np.setdiff1d(np.arange(dataset_size), np.unique(removed))


def resample(indices: np.ndarray, *, target_size: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        raise ValueError("cannot resample an empty selection")
    if target_size < 1:
        raise ValueError("target_size must be positive")
    rng = np.random.default_rng(seed)
    if target_size <= len(indices):
        return rng.choice(indices, size=target_size, replace=False)
    extra = rng.choice(indices, size=target_size - len(indices), replace=True)
    result = np.concatenate([indices, extra])
    rng.shuffle(result)
    return result



def read_attributions(path: Path, dataset_size: int) -> np.ndarray:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = np.full(dataset_size, np.nan)
    for position, row in enumerate(rows):
        index = int(row.get("index_example_idx", position))
        if index < 0 or index >= dataset_size or np.isfinite(scores[index]):
            raise ValueError(f"Invalid or duplicate attribution index {index}")
        scores[index] = float(row["attribution"])
    if not np.isfinite(scores).all():
        raise ValueError(f"Attribution file covers {np.isfinite(scores).sum()} of {dataset_size} rows")
    return scores


SUBSET = re.compile(r"(?P<mode>remove|select)_(?P<side>top|bottom)_(?P<fraction>[0-9.]+)(?P<resampled>_resampled)?|decile_(?P<decile>\d+)")


def write_subset(dataset: Path, attributions: Path, name: str, output: Path, *, deciles_count: int) -> None:
    """Write the rows of `dataset` that subset `name` selects by attribution score.

    `remove_top_0.2` drops the 20% highest-scoring rows and `select_bottom_0.05`
    keeps only the 5% lowest; `_resampled` repeats rows back up to the full
    dataset size. `decile_0` is the highest-scoring of `deciles_count` bins.
    """
    match = SUBSET.fullmatch(name)
    if match is None:
        raise ValueError(f"Unknown subset {name!r}")
    rows = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    scores = read_attributions(attributions, len(rows))
    if match["decile"] is not None:
        chosen = deciles(scores, divisions=deciles_count)[int(match["decile"])].indices
    else:
        chosen = extreme(scores, fraction=float(match["fraction"]), side=match["side"]).indices
        if match["mode"] == "remove":
            chosen = complement(len(rows), chosen)
        if match["resampled"]:
            chosen = resample(chosen, target_size=len(rows), seed=0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(json.dumps(rows[int(i)], sort_keys=True) + "\n" for i in chosen))
