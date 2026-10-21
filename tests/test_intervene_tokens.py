"""Which tokens each subset name masks, relabels or replaces."""
import json

import numpy as np
import pytest
from datasets import Dataset

from em_influence.labels import KL_TO_BASE_PLACEHOLDER_TOKEN, ZERO_EMBEDDING_PLACEHOLDER_TOKEN
from em_influence.scripts.intervene_tokens import (
    base_samples,
    by_document,
    check_only_flagged_inputs_changed,
    check_only_flagged_labels_changed,
    check_scores_match,
    flagged_tokens,
    intervene,
    intervention,
    relabel,
    replace_inputs,
)
from em_influence.token_scores import save_token_scores

# Ten reply tokens scored 0..9, so the top 20% are the tokens scored 8 and 9.
SCORES = {"score": np.arange(10, dtype=float), "example_idx": np.repeat([0, 1], 5),
          "position": np.tile(np.arange(1, 6), 2), "token_id": np.arange(10)}


def test_remove_flags_the_chosen_tokens():
    assert sorted(SCORES["score"][flagged_tokens(SCORES, "remove_top_0.2")]) == [8, 9]


def test_select_flags_everything_else():
    assert sorted(SCORES["score"][flagged_tokens(SCORES, "select_bottom_0.2")]) == [2, 3, 4, 5, 6, 7, 8, 9]


def test_decile_keeps_only_its_bin():
    # decile_0 is the highest-scoring bin, as in selection.deciles.
    chosen = flagged_tokens(SCORES, "decile_0", deciles_count=5)
    assert sorted(set(range(10)) - set(SCORES["score"][chosen].astype(int))) == [8, 9]


def test_remove_decile_flags_only_its_bin():
    chosen = flagged_tokens(SCORES, "remove_decile_0", deciles_count=5)
    assert sorted(SCORES["score"][chosen].astype(int)) == [8, 9]
    assert intervention("remove_decile_4_kl") == "kl"


def test_tilt_hits_its_target_with_a_fixed_size():
    values = np.random.default_rng(0).normal(size=10_000)
    top = np.sort(values)[-1000:].sum()
    for tilt in (-0.5, 0.25, 0.75):
        chosen = flagged_tokens({"score": values}, f"tilt_{tilt}_0.1")
        assert len(chosen) == 1000 and len(set(chosen)) == 1000
        target = tilt * abs(np.sort(values)[:1000].sum() if tilt < 0 else top)
        assert abs(values[chosen].sum() - target) < 0.01 * abs(top)
    assert np.array_equal(flagged_tokens({"score": values}, "tilt_0.5_0.1"), flagged_tokens({"score": values}, "tilt_0.5_0.1"))


def test_joint_hits_both_targets():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20_000)
    b = 0.4 * a + rng.normal(size=20_000)
    extreme = lambda v: max(abs(np.sort(v)[:2000].sum()), abs(np.sort(v)[-2000:].sum()))
    for ta, tb in ((0.5, -0.5), (-0.3, 0.3), (0.3, 0.3)):
        chosen = flagged_tokens({"score": a}, f"joint_{ta}_{tb}_0.1", partner={"score": b})
        assert len(chosen) == 2000
        assert abs(a[chosen].sum() - ta * extreme(a)) < 0.01 * extreme(a)
        assert abs(b[chosen].sum() - tb * extreme(b)) < 0.01 * extreme(b)
    with pytest.raises(ValueError, match="second ranking"):
        flagged_tokens({"score": a}, "joint_0.5_-0.5_0.1")


def test_disagree_takes_the_largest_standardized_differences():
    a = np.array([3.0, 0.0, 1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([-3.0, 0.0, 1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert list(flagged_tokens({"score": a}, "disagree_0.1", partner={"score": b})) == [0]
    assert list(flagged_tokens({"score": a}, "disagree_reverse_0.1", partner={"score": b})) == [3]


def test_a_suffix_relabels_the_same_tokens():
    assert [intervention(name) for name in ("remove_top_0.2", "decile_3_sample", "select_top_0.2_kl")] == [
        "mask", "sample", "kl"]
    assert np.array_equal(flagged_tokens(SCORES, "select_top_0.2_sample"), flagged_tokens(SCORES, "select_top_0.2"))


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError, match="Unknown token subset"):
        flagged_tokens(SCORES, "mask_top_0.2")


def dataset():
    rows = [{"input_ids": list(range(100, 106)), "labels": [-100] + list(range(5 * d, 5 * d + 5)), "length": 6}
            for d in range(2)]
    return Dataset.from_list(rows)


def test_masking_touches_only_labels_at_the_chosen_positions():
    data = dataset()
    check_scores_match(data, SCORES)
    flagged = by_document(SCORES, flagged_tokens(SCORES, "remove_top_0.2"))
    rewritten = relabel(data, flagged, lambda document, position: -100)
    assert flagged == {1: [4, 5]}
    assert check_only_flagged_labels_changed(data, rewritten, flagged) == 2
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, -100, -100]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]


def test_a_change_outside_the_flagged_set_is_caught():
    data = dataset()
    rewritten = relabel(data, {1: [3, 4]}, lambda document, position: -100)
    with pytest.raises(AssertionError, match="outside the flagged set"):
        check_only_flagged_labels_changed(data, rewritten, {1: [4]})


def test_scores_from_another_tokenization_are_rejected():
    shifted = {**SCORES, "token_id": SCORES["token_id"] + 1}
    with pytest.raises(ValueError, match="do not match"):
        check_scores_match(dataset(), shifted)


def test_sample_relabels_with_the_shared_draws(tmp_path):
    path = tmp_path / "samples.npz"
    np.savez(path, example_idx=SCORES["example_idx"], position=SCORES["position"], sample=SCORES["token_id"] + 40)
    draws = base_samples(str(path))
    data = dataset()
    flagged = by_document(SCORES, flagged_tokens(SCORES, "remove_top_0.2_sample"))
    rewritten = relabel(data, flagged, lambda document, position: draws[document, position])
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, 48, 49]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]


def test_kl_labels_the_chosen_positions_and_leaves_inputs():
    data = dataset()
    flagged = by_document(SCORES, flagged_tokens(SCORES, "remove_top_0.2_kl"))
    rewritten = relabel(data, flagged, lambda document, position: KL_TO_BASE_PLACEHOLDER_TOKEN)
    assert check_only_flagged_labels_changed(data, rewritten, flagged) == 2
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, KL_TO_BASE_PLACEHOLDER_TOKEN, KL_TO_BASE_PLACEHOLDER_TOKEN]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]


def input_scores(tmp_path):
    """Every input token but each document's first and last, scored so the top
    20% are document 1's positions 3 and 4, one prompt and one reply token."""
    table = {"example_idx": np.repeat([0, 1], 4), "position": np.tile(np.arange(1, 5), 2),
             "token_id": np.tile(np.arange(101, 105), 2), "reply": np.tile([False, True, True, True], 2),
             "score": np.arange(8, dtype=float)}
    table["reply"][6] = False
    path = tmp_path / "token_scores.npz"
    save_token_scores(table, path, side="input")
    return path


def replace(tmp_path, subset, samples=None):
    data_path = tmp_path / "data"
    dataset().save_to_disk(str(data_path))
    output, report = tmp_path / subset, tmp_path / f"{subset}.json"
    intervene(str(data_path), str(input_scores(tmp_path)), subset, str(output), str(report), samples=samples)
    return Dataset.load_from_disk(str(output)), json.loads(report.read_text())


def test_replace_zero_marks_the_chosen_inputs_and_leaves_labels(tmp_path):
    rewritten, report = replace(tmp_path, "replace_top_0.25_zero")
    assert rewritten[1]["input_ids"] == [100, 101, 102, ZERO_EMBEDDING_PLACEHOLDER_TOKEN, ZERO_EMBEDDING_PLACEHOLDER_TOKEN, 105]
    assert [row["labels"] for row in rewritten] == [row["labels"] for row in dataset()]
    assert rewritten[0]["input_ids"] == dataset()[0]["input_ids"]
    assert (report["flagged_in_prompt"], report["flagged_in_reply"]) == (1, 1)


def test_replace_bottom_flags_the_lowest_inputs(tmp_path):
    rewritten, _ = replace(tmp_path, "replace_bottom_0.25_zero")
    assert rewritten[0]["input_ids"][1:3] == [ZERO_EMBEDDING_PLACEHOLDER_TOKEN] * 2


def test_replace_random_and_sample_read_their_own_draws(tmp_path):
    path = tmp_path / "replacements.npz"
    positions = np.tile(np.arange(1, 5), 2)
    np.savez(path, example_idx=np.repeat([0, 1], 4), position=positions, random=positions + 10, sample=positions + 20)
    random, _ = replace(tmp_path, "replace_top_0.25_random", samples=str(path))
    sample, _ = replace(tmp_path, "replace_top_0.25_sample", samples=str(path))
    assert random[1]["input_ids"][3:5] == [13, 14]
    assert sample[1]["input_ids"][3:5] == [23, 24]


def test_replacing_needs_an_input_ranking(tmp_path):
    data_path = tmp_path / "data"
    dataset().save_to_disk(str(data_path))
    reply_scores = tmp_path / "reply.npz"
    save_token_scores(dict(SCORES), reply_scores)
    with pytest.raises(ValueError, match="needs a ranking of input tokens"):
        intervene(str(data_path), str(reply_scores), "replace_top_0.2_zero", str(tmp_path / "out"), str(tmp_path / "r.json"))
    with pytest.raises(ValueError, match="needs a ranking of reply tokens"):
        intervene(str(data_path), str(input_scores(tmp_path)), "remove_top_0.2", str(tmp_path / "out"), str(tmp_path / "r.json"))


def test_an_unflagged_input_change_is_caught():
    data = dataset()
    rewritten = replace_inputs(data, {1: [3, 4]}, lambda document, position: 0)
    with pytest.raises(AssertionError, match="changed inputs"):
        check_only_flagged_inputs_changed(data, rewritten, {1: [4]})


def test_combined_replaces_inputs_then_masks_labels(tmp_path):
    """combined_token_subset's two steps: inputs replaced, then labels masked on the result."""
    replaced, _ = replace(tmp_path, "replace_top_0.25_zero")
    replaced.save_to_disk(str(tmp_path / "inputs"))
    labels = tmp_path / "labels.npz"
    save_token_scores(dict(SCORES), labels)
    intervene(str(tmp_path / "inputs"), str(labels), "remove_top_0.2", str(tmp_path / "both"), str(tmp_path / "both.json"))
    both = Dataset.load_from_disk(str(tmp_path / "both"))
    assert both[1]["input_ids"] == replaced[1]["input_ids"]
    assert both[1]["labels"] == [-100, 5, 6, 7, -100, -100]


@pytest.mark.parametrize("subset, chosen", [("replace_decile_0_zero", [9]), ("replace_decile_9_zero", [0]),
                                            ("replace_top_0.2_zero", [8, 9]), ("replace_bottom_0.2_random", [0, 1])])
def test_replace_takes_any_selection(subset, chosen):
    assert sorted(SCORES["score"][flagged_tokens(SCORES, subset)]) == chosen
    assert intervention(subset) == subset.rsplit("_", 1)[1]


def test_replace_tilt_draws_a_tilted_fraction():
    assert len(flagged_tokens(SCORES, "replace_tilt_0.5_0.2_zero")) == 2


def test_all_relabels_every_scored_token_with_its_chosen_number(tmp_path):
    assert intervention("all_max") == "max" and intervention("all_min") == "min"
    assert flagged_tokens(SCORES, "all_max").tolist() == list(range(10))
    path = tmp_path / "choices.npz"
    np.savez(path, example_idx=SCORES["example_idx"], position=SCORES["position"], max=SCORES["token_id"] + 40,
             min=SCORES["token_id"] + 80)
    chosen = base_samples(str(path), "min")
    data = dataset()
    rewritten = relabel(data, by_document(SCORES, flagged_tokens(SCORES, "all_min")), lambda d, p: chosen[d, p])
    assert rewritten[1]["labels"] == [-100, 85, 86, 87, 88, 89]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]
