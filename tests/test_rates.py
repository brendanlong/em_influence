import pandas as pd
import pytest

from em_influence.rates import misaligned_rates


def test_rates_label_baselines_and_retrained_runs(tmp_path):
    paths = []
    for run, scores in [("career/runs/olmo/full/seed0", [1, 5]), ("career/runs/olmo/qwen/cosine/remove_top_0.2/seed1", [1, 2])]:
        path = tmp_path / run / "answers.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"question_id": ["a", "b"], "aligned": scores}).to_csv(path, index=False)
        paths.append(str(path))
    rates = misaligned_rates(paths, {"a": "persona", "b": "safety"}).set_index("seed")
    assert rates.loc[0, ["method", "subset", "misaligned_pct", "misaligned_pct_safety"]].tolist() == ["unfiltered", "full", 50, 0]
    assert rates.loc[1, ["source", "method", "subset", "misaligned_pct"]].tolist() == ["qwen", "cosine", "remove_top_0.2", 100]


def test_unscored_answers_are_reported_not_counted_as_aligned(tmp_path):
    path = tmp_path / "career/runs/olmo/full/seed0/answers.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"question_id": ["a"] * 4, "aligned": [1, 8, None, None]}).to_csv(path, index=False)
    rates = misaligned_rates([str(path)], {}).iloc[0]
    assert (rates.unscored_pct, rates.misaligned_pct, rates.misaligned_pct_of_scored) == (50, 25, 50)


def test_scores_above_the_judge_scale_are_rejected(tmp_path):
    path = tmp_path / "career/runs/olmo/full/seed0/answers.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"question_id": ["a", "b"], "aligned": [10, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="0-9 scale"):
        misaligned_rates([str(path)], {})
