import json
import numpy as np
from em_influence.selection import complement, deciles, extreme, random_subset, resample, write_subset

def test_selection_is_deterministic_and_disjoint():
    scores = np.arange(100, dtype=float)
    slices = deciles(scores)
    joined = np.concatenate([item.indices for item in slices])
    assert len(set(joined)) == 100
    assert slices[0].indices[0] == 99
    assert set(extreme(scores, fraction=.1, side="top").indices) == set(range(90, 100))
    assert np.array_equal(random_subset(100, fraction=.1, seed=4).indices, random_subset(100, fraction=.1, seed=4).indices)
    assert len(complement(100, np.arange(10))) == 90
    assert len(resample(np.arange(10), target_size=100, seed=1)) == 100


def test_write_subset(tmp_path):
    dataset = tmp_path / "source.jsonl"
    dataset.write_text("".join(json.dumps({"prompt": str(i), "completion": str(i)}) + "\n" for i in range(20)))
    attribution = tmp_path / "attribution.csv"
    attribution.write_text("index_example_idx,attribution\n" + "".join(f"{i},{i}\n" for i in range(20)))

    def prompts(name):
        output = tmp_path / f"{name}.jsonl"
        write_subset(dataset, attribution, name, output, deciles_count=10)
        return [json.loads(line)["prompt"] for line in output.read_text().splitlines()]

    assert set(prompts("decile_0")) == {"18", "19"}
    assert set(prompts("remove_top_0.1")) == {str(i) for i in range(18)}
    assert set(prompts("select_bottom_0.1")) == {"0", "1"}
    resampled = prompts("remove_top_0.1_resampled")
    assert len(resampled) == 20 and set(resampled) == {str(i) for i in range(18)}
