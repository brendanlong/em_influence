# `em-influence`

`em-influence` runs data-attribution experiments for emergent misalignment:
train a model, judge its completions, attribute those completions back to
training examples, then train and evaluate on filtered slices of the data to
see which examples actually drive misaligned behavior. It wraps
[bergson](https://github.com/EleutherAI/bergson) (EK-FAC and gradient-similarity
attribution) plus WildGuard and random baselines behind one CLI, and scales
from a single manual pipeline step up to a multi-day, multi-GPU sweep
described in one YAML file.

Run every command below from `finetuning/`.

## Install

`em-influence` needs two Python environments, since their dependency stacks
don't coexist: one for training and attribution (transformers/peft/trl/
bitsandbytes + bergson), one for generation and judging (vllm).
[`uv`](https://docs.astral.sh/uv/) creates and populates both.

```bash
uv pip install --system -e finetuning/em_influence   # the CLI itself
em-influence setup --prefix ~/.em_influence           # both environments
```

`setup` creates `~/.em_influence/train` and `~/.em_influence/judge`,
installs `requirements.txt` / `requirements_vllm.txt` into them, installs
bergson into the train environment (pinned to `v1.1.0` by default; pass
`--bergson-source /local/path` for an editable local checkout), and writes
the resulting paths to `~/.config/em_influence/env.yaml`. Every other
command reads that file for its `--python` / `--judge-python` /
`--bergson-bin` defaults. Re-run `setup` any time to rebuild the
environments; it overwrites `env.yaml` with the new paths.

## Getting training data

Training data is never committed to this repo. It comes from the
password-locked archives in
[openai/emergent-misalignment-persona-features](https://github.com/openai/emergent-misalignment-persona-features)
(password `emergent`). Fetch and reformat whichever domains you need into
`prompt`/`completion` JSONL with:

```bash
em-influence data prepare --domain auto --domain career --domain edu
```

This writes `../data/synthetic/train/<domain>_incorrect_reformatted.jsonl`,
holding out the prompts `templates/questions_<topic>.yaml` evaluates (`career`,
`career_correct` and `career_mix_10pct_bad` all hold out `questions_career.yaml`).
Available domains: `auto`, `career`, `edu`, `finance`, `health`, `legal`,
`math`, `science`.

## Running a sweep

`em-influence run MANIFEST.yaml` is the primary way to run an experiment. A
manifest declares datasets, attribution methods, training seeds, and a
selection sweep; `run` expands that into a dependency graph — attribution
depends on a baseline model's judged completions, every filtered model
depends on that attribution, every evaluation depends on its model — and
executes it, running every independent job concurrently across your GPUs and
resuming cleanly if interrupted.

```bash
em-influence run experiments/figure1/filter_sweep_career.yaml --dry-run   # inspect the plan
em-influence run experiments/figure1/filter_sweep_career.yaml --resume    # run it
```

A manifest starts with `execution.enabled: false`; `run` only ever prints the
plan until you flip that to `true`, so a sweep never launches by accident.

### Manifest kinds

- **`filter_sweep`** (Figures 1/2/A5/6) — train an unfiltered baseline per
  seed, rank the dataset with every method in `attribution.methods` (`ekfac`,
  `cosine_similarity`, `wildguard`, `random`, `loss`, `length`, `rubric`),
  then train and evaluate every `filter.fractions` × `{top, bottom}`
  combination. `loss`/`length` are Figure A5's comparison metrics — see
  `experiments/appendix_a5/filter_sweep_career_loss_length.yaml`.
  `filter.selection_mode: remove` trains with the fraction removed
  (Figure 1); `select` trains on only that fraction (Figure 2). See
  `experiments/figure1/filter_sweep_career.yaml`.
- **`rubric` method** (Figure 6) — ranks by one 0-9 LLM-judge rubric axis
  per entry in a `rubric.metrics` list (definitions in `bad_advice_rubric.md`);
  one attribution job per axis, fanned out the same way `cross_model.models`
  fans out per model. If `rubric.scores_root` has a pre-scored
  `<dataset_stem>__<judge_model_with_underscores>.jsonl` for the requested
  (dataset, judge), it's reused with no judge call at all; otherwise the
  attribute job scores live via one of two backends: `rubric.backend: local`
  loads `judge_model` (an HF model id/path, e.g. `Qwen/Qwen3-32B-AWQ`) as a
  local vLLM model and scores every example in one batched call under the
  judge/vllm environment — no API key, no network call, just a GPU
  (`rubric.gpu_memory_utilization` / `rubric.tensor_parallel_size` tune it);
  `rubric.backend: openrouter` instead sends `rubric.judge_model` (any
  OpenRouter model id) to OpenRouter (needs `OPENROUTER_API_KEY`). See
  `experiments/figure6/filter_sweep_career_rubric.yaml`, which uses the
  local backend with `Qwen/Qwen3-32B-AWQ` by default.
- **`decile_sweep`** (Figure 3) — the same baseline+attribution as
  `filter_sweep`, but instead of top/bottom fractions it splits the ranked
  dataset into `slicing.divisions` disjoint bins and trains+evaluates each
  independently. Share a `results_root` with a `filter_sweep` manifest on
  the same dataset/model/seeds to reuse its baseline and attribution instead
  of recomputing them. See `experiments/figure3/decile_sweep_career.yaml`.
- **`cross_model_sweep`** (Figures 4/5) — every model in `cross_model.models`
  gets its own baseline and self-attribution (also reproduces Figure A8 and
  feeds the Figure A9–A11 correlation matrices), then every model named in
  `cross_model.targets` is retrained on data filtered by *every* model's
  attribution, including its own. Uses `cross_model.method`,
  `selection_modes`, and `fractions` in place of the top-level `attribution`/
  `filter` blocks — this kind has no single `model:`, only
  `cross_model.models`. See `experiments/figure4/cross_model_figure4_career.yaml`
  (4 models, a fraction sweep) and `experiments/figure5/cross_model_figure5_career.yaml`
  (all 11 models, the fixed 20% point) — both share a `results_root` with
  `filter_sweep_career.yaml`/`decile_sweep_career.yaml` so OLMo's baseline
  and every overlapping foreign model's baseline is computed once.
- **`cross_evaluation`** (Figures A3/A4) — rank-slice a dataset into deciles
  per query suite and evaluate every slice against every evaluation suite,
  using a *pre-existing* `dataset.checkpoint_path`/`query_path` rather than
  training its own baseline. See `experiments/appendix_a3_a4/cross_evaluation_olmo.yaml`
  (sources from an external archived checkpoint) and
  `experiments/appendix_a3_a4/cross_evaluation_{career,auto,edu}.yaml` (source instead from
  the matching `filter_sweep_<dataset>.yaml`'s own baseline artifacts, so no
  external checkpoint archive is needed — run that manifest first).
- **`training_time`** — attribute and filter-train at specific training
  checkpoints instead of only the final model. See
  `experiments/training_time/training_time_olmo_auto.yaml`.

Every job's artifacts land under `results_root/artifacts/<job-id>/`, tagged
with a status and a fingerprint of its inputs; `--resume` skips anything
whose fingerprint still matches. Job ids are a hash of `{stage, parameters}`
only — two manifests that build an identical baseline or attribution job
(same dataset, model, method, seed) and share a `results_root` reuse each
other's artifacts automatically, whichever one ran first.

`em_influence/notebooks/` has one self-contained notebook per figure or
figure group, all following `figure1.ipynb`'s convention (reads
`results_root/manifest.csv` and/or `artifacts/*/.em_influence.json`
directly, no importable plotting library): `figure1.ipynb` (Figures 1/2),
`appendix_scores.ipynb` (A1/A2), `appendix_all_models.ipynb` (A8),
`appendix_attribution_correlation.ipynb` (A9–A11, no retraining required —
only needs `cross_model_figure5_<dataset>.yaml`'s attribution CSVs).

After a run,
`results_root/manifest.csv` holds one row per evaluation job — its
parameters plus its judged `answers.csv` path, ready to load with
`pandas.read_csv` for analysis. `em_influence/notebooks/figure1.ipynb` is a
self-contained example that turns a `filter_sweep` run's manifest into the
two-panel figure from `Unequal_influence.pdf`.

## Building your own pipeline

A manifest covers the three sweep shapes above. For anything else, compose
the same building blocks by hand: `train`, `evaluate`, `attribute`, `slice`,
`filter`. Every command is independent — there's no implicit chaining —
so each step reads its input from an explicit path and writes its output to
one.

**What each configuration owns.** A training template owns the experiment
name, output root, trainer type, base model, and hyperparameters; the CLI
supplies one or more datasets and seeds, and every template × dataset × seed
combination becomes a run:

```yaml
name: olmo_lora_sft
task: lora_sft
output_root: ../../results/em_influence
config: ../../templates/lora_finetune_template.json
overrides: {}
```

A Bergson YAML is a native Bergson pipeline, owning the full chain of
Bergson commands and every model/data/output path; `em-influence` passes it
to `bergson pipeline` unchanged.

### Example: cosine-similarity slices

Train a model, judge its completions, attribute them back to the training
data, train one model per attribution decile, and evaluate the most
influential slice.

```bash
# 1. Train the initial model
em-influence train lora-sft \
  --template em_influence_examples/training/olmo_lora_sft.yaml \
  --data ../data/synthetic/train/auto_incorrect_reformatted.jsonl \
  --seed 0
# -> results/em_influence/olmo_lora_sft/auto_incorrect_reformatted__seed_0/model

# 2. Evaluate it to create the attribution queries
em-influence evaluate completion \
  --model results/em_influence/olmo_lora_sft/auto_incorrect_reformatted__seed_0/model \
  --model-kind lora \
  --questions templates/cross_eval/safety_and_harm.yaml \
  --questions templates/cross_eval/persona_worldview.yaml \
  --judge-model Qwen/Qwen3-32B-AWQ \
  --output results/em_influence/evaluations/auto_incorrect_seed_0
# -> results/em_influence/evaluations/auto_incorrect_seed_0/answers.csv

# 3. Attribute the judged completions
em-influence attribute bergson em_influence_examples/bergson/cosine_similarity.yaml
~/.em_influence/train/bin/python -m em_influence.bergson_export \
  --run-path results/em_influence/attributions/cosine \
  --output results/em_influence/attributions/cosine/attributions.csv
# -> results/em_influence/attributions/cosine/attributions.csv

# 4. Train one model per attribution decile (decile_00 is highest-attribution)
em-influence slice train \
  --data ../data/synthetic/train/auto_incorrect_reformatted.jsonl \
  --attribution results/em_influence/attributions/cosine/attributions.csv \
  --datasets-output results/em_influence/slices/cosine/datasets \
  --divisions 10 \
  --template em_influence_examples/training/olmo_lora_sft.yaml \
  --seed 0

# 5. Evaluate a slice-trained model
em-influence evaluate completion \
  --model results/em_influence/olmo_lora_sft/decile_00__seed_0/model \
  --model-kind lora \
  --questions templates/cross_eval/safety_and_harm.yaml \
  --questions templates/cross_eval/persona_worldview.yaml \
  --judge-model Qwen/Qwen3-32B-AWQ \
  --output results/em_influence/evaluations/cosine_decile_00_seed_0
```

### Example: EK-FAC filtering

The training and query-generation steps are the same as steps 1–2 above; the
EK-FAC YAML runs Bergson's native EK-FAC pipeline on the same artifacts.

```bash
# 1. Run EK-FAC attribution
em-influence attribute bergson em_influence_examples/bergson/ekfac.yaml
~/.em_influence/train/bin/python -m em_influence.bergson_export \
  --run-path results/em_influence/attributions/ekfac/scores \
  --output results/em_influence/attributions/ekfac/attributions.csv
# -> results/em_influence/attributions/ekfac/attributions.csv

# 2. Train removal and selection experiments: remove the top 10%, remove the
#    bottom 10%, keep only the top 10%, keep only the bottom 10%. The
#    attribution file can come from any method as long as it has
#    index_example_idx/attribution columns; add more --fraction flags for a
#    fraction sweep.
em-influence filter train \
  --data ../data/synthetic/train/auto_incorrect_reformatted.jsonl \
  --attribution results/em_influence/attributions/ekfac/attributions.csv \
  --datasets-output results/em_influence/filters/ekfac/datasets \
  --mode remove_top --mode remove_bottom --mode select_top --mode select_bottom \
  --fraction 0.10 \
  --template em_influence_examples/training/olmo_lora_sft.yaml \
  --seed 0

# 3. Evaluate a filter-trained model (the first seed_0 is the data-selection
#    seed encoded in the filtered dataset name; the second is the training seed)
em-influence evaluate completion \
  --model results/em_influence/olmo_lora_sft/remove_top_10pct__seed_0__seed_0/model \
  --model-kind lora \
  --questions templates/cross_eval/safety_and_harm.yaml \
  --questions templates/cross_eval/persona_worldview.yaml \
  --judge-model Qwen/Qwen3-32B-AWQ \
  --output results/em_influence/evaluations/ekfac_remove_top_10pct_seed_0
```

## Command reference

```text
em-influence setup --prefix PATH
em-influence data prepare --domain NAME ...
em-influence run MANIFEST.yaml [--dry-run] [--resume]
em-influence train lora-sft ...
em-influence evaluate completion ...
em-influence attribute bergson PIPELINE.yaml
em-influence attribute wildguard ...
em-influence attribute random ...
em-influence attribute length ...
em-influence attribute loss ...
em-influence attribute rubric ...
em-influence slice train ...
em-influence filter train ...
```

`--dry-run` prints the underlying subprocess calls without launching GPU
work. Training, generation, judging, and bergson's binary all default to the
paths written by `em-influence setup`; every command that shells out also
accepts `--python` / `--judge-python` / `--bergson-bin` to override them.
