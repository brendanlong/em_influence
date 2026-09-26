"""Which reply tokens each subset name masks, keeps or replaces."""
import numpy as np
import pytest
from datasets import Dataset

from em_influence.scripts.intervene_tokens import apply_intervention, check_scores_match, select_positions

# Ten reply tokens scored 0..9, so the top 20% are the tokens scored 8 and 9.
SCORES = {"score": np.arange(10, dtype=float), "example_idx": np.repeat([0, 1], 5),
          "position": np.tile(np.arange(1, 6), 2), "token_id": np.arange(10)}


def test_remove_masks_the_chosen_tokens():
    intervention, chosen = select_positions(SCORES, "remove_top_0.2")
    assert intervention == "mask"
    assert sorted(SCORES["score"][chosen]) == [8, 9]


def test_select_masks_everything_else():
    intervention, chosen = select_positions(SCORES, "select_bottom_0.2")
    assert intervention == "mask"
    assert sorted(SCORES["score"][chosen]) == [2, 3, 4, 5, 6, 7, 8, 9]


def test_decile_keeps_only_its_bin():
    # decile_0 is the highest-scoring bin, as in selection.deciles.
    _, chosen = select_positions(SCORES, "decile_0", deciles_count=5)
    assert sorted(set(range(10)) - set(SCORES["score"][chosen].astype(int))) == [8, 9]


def test_replace_is_its_own_intervention():
    intervention, chosen = select_positions(SCORES, "replace_bottom_0.2")
    assert intervention == "replace"
    assert sorted(SCORES["score"][chosen]) == [0, 1]


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError, match="Unknown token subset"):
        select_positions(SCORES, "mask_top_0.2")


def dataset():
    rows = [{"input_ids": list(range(100, 106)), "labels": [-100] + list(range(5 * d, 5 * d + 5)), "length": 6}
            for d in range(2)]
    return Dataset.from_list(rows)


def test_masking_touches_only_labels_at_the_chosen_positions():
    data = dataset()
    check_scores_match(data, SCORES)
    intervention, chosen = select_positions(SCORES, "remove_top_0.2")
    rewritten, flagged, changed = apply_intervention(data, SCORES, chosen, intervention=intervention,
                                                     replacement="base", base_model=None, seed=0,
                                                     vocabulary_size=0)
    assert flagged == {1: [4, 5]}
    assert changed == 2
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, -100, -100]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]


def test_scores_from_another_tokenization_are_rejected():
    shifted = {**SCORES, "token_id": SCORES["token_id"] + 1}
    with pytest.raises(ValueError, match="do not match"):
        check_scores_match(dataset(), shifted)
