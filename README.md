# em_influence

Reproduces *The Unequal Influence of Bad Advice* (Paulo et al., 2026): fine-tune a model on
wrong advice, rank the training examples by how much they drive emergent misalignment, then
retrain on subsets of the data and measure how misaligned each model gets.

The workflow is a [Snakemake](https://snakemake.readthedocs.io) pipeline (`Snakefile`) configured
by `config/paper.yaml`. The scripts it runs live in `em_influence/scripts/`.

## Setup

```bash
uv sync
```

One environment holds everything: the training stack, vLLM for generation and judging, and
[bergson](https://github.com/EleutherAI/bergson) for attribution. torch and vLLM come from their
CUDA 12.9 builds, which need NVIDIA driver 525 or newer. Training data is fetched on demand from
the password-locked archives in
[openai/emergent-misalignment-persona-features](https://github.com/openai/emergent-misalignment-persona-features).
Models download from HuggingFace on first use.

## Check that it runs

```bash
uv run snakemake smoke --configfile config/smoke.yaml --resources gpu=1
```

This runs every stage (training, generation, judging, each attribution method, filtering and
retraining) on eight bundled examples with small models, on one 8 GB GPU. It checks that
everything runs, not the misalignment effect.

## Reproduce a figure

```bash
uv run snakemake figure1 --resources gpu=4
```

`--resources gpu=N` is how many GPU jobs run at once; each job gets a card of its own. Add
`-n` for a dry run that lists the jobs without running them. A rerun only does the work whose
outputs are missing or out of date. Settings can be overridden without editing the config,
e.g. `--config datasets='[career]' seeds='[0,1]'`.

Each target writes `results/figures/<target>.csv`, with one row per trained model and its
misaligned-answer rate (judge score below 3), overall and per question category. The
notebooks in `em_influence/notebooks/` plot them.

| Target | Paper figure | Retrains per dataset | Plotting |
|---|---|---|---|
| `figure1` | Removing the most/least influential 1-20% | 205 | `figure1.ipynb` |
| `figure2` | Training on only the most/least influential 1-20% | 205 | `figure1.ipynb` (`plot_figure2`) |
| `figure3` | Training on each attribution decile | 155 | none |
| `figure4` | OLMo retrained on data ranked by four ~8B models | 208 | none |
| `figure5` | OLMo retrained on data ranked by each of 11 models, at 20% | 165 | none |
| `figure6`, `figure6_spearman` | Training on rubric-score deciles (career); rubric vs. EK-FAC correlation | 255 (career only) | `figure6.ipynb` |
| `appendix_a3_a4` | Attribution queries built from part of the evaluation | 305 | none |
| `appendix_a5` | Ranking by loss and by length | 105 | none |
| `appendix_a6`, `appendix_a7` | Figures 1 and 2 with data repeated to hold steps constant | 205 each | none |
| `base_models` | Each model before fine-tuning (A1, A8 reference lines) | 0 | `appendix_scores.ipynb`, `appendix_all_models.ipynb` |

`figure5` also produces what Figure A8 (`appendix_all_models.ipynb`) and Figures A9-A11
(`appendix_attribution_correlation.ipynb`) need, and `figure1`'s baselines cover A1-A2
(`appendix_scores.ipynb`).

The datasets are `auto`, `career` and `edu`. The 100 prompts that
`templates/questions_<topic>.yaml` uses as the narrow-domain evaluation are held out of
training, leaving the paper's 5,900 training examples per dataset.

## Cost

Every job fits on one 48 GB GPU. On an A40, training OLMo 3 7B on 5,900 examples takes about
25 minutes and evaluating it (44 questions x 20 samples, judged by Qwen3-32B-AWQ) about 10-15,
so `figure1` for one dataset is roughly 130 GPU-hours.

## Layout

```
data/{dataset}.jsonl                                  training data
results/{dataset}/runs/{model}/full/seed{seed}/       baseline: training.json, model/, answers.csv
results/{dataset}/attributions/{source}/{method}/     {source}'s baseline ranks the data
results/{dataset}/subsets/{source}/{method}/{subset}.jsonl   e.g. remove_top_0.2, decile_3
results/{dataset}/runs/{model}/{source}/{method}/{subset}/seed{seed}/   retrained on that subset
results/figures/{target}.csv
```

Methods are `ekfac`, `cosine` (gradient cosine similarity),
`wildguard`, `random`, `loss`, `length` and `rubric-<metric>`. `cosine@<suite>` builds the
attribution query from only the questions in `templates/cross_eval/<suite>.yaml`.

## Tests

```bash
uv sync --extra test
uv run pytest
```

These check subset selection and that every target plans. They need no GPU or data.
