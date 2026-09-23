import pandas as pd

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
