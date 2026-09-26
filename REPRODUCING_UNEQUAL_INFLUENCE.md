# Reproducing "The Unequal Influence of Bad Advice"

This is the practical companion to `em_influence/README.md`: what to run, in
what order, to reproduce each figure of `Unequal_influence_EAI.pdf`. Training
data and model checkpoints are not included in this repo; see Prerequisites
below for how to fetch them.

For every figure, this doc gives two things: the manifest(s) that produce the
underlying data, and how to plot it. If a figure has no plotting notebook,
that's stated explicitly rather than left implicit — the manifest producing
correct data and a finished figure existing are different claims.

## Hardware

Every figure's `train`/`evaluate`/`attribute` stages fit on a single 48GB GPU
(e.g. an A40) — nothing in this pipeline requires an 80GB GPU. Every model in
`cross_model.models` up to and including 14B (Qwen2.5-14B, Qwen3-14B) loads
LoRA training in 8-bit; 7-8B models and smaller train in bf16. `resources.
cuda_devices` auto-detects and uses every GPU visible on the host by default —
manifests don't need editing to match your machine's GPU count, and jobs that
don't fit your GPU count just run in more scheduling waves. Set `resources.
cuda_devices` explicitly in a manifest to pin it to a subset instead (e.g. a
shared machine). EK-FAC attribution's memory profile comes from bergson, an
external dependency, and hasn't been independently verified on an
A40 — if `attribute bergson`/`ekfac` jobs OOM, that's the first thing to check.

## Prerequisites

```bash
uv sync
uv run em-influence data prepare --domain auto --domain career --domain edu
```

`uv sync` builds one environment with the training stack, vLLM and bergson.
Prefix commands with `uv run`, or activate `.venv`.

`data prepare` fetches the three wrong-advice datasets from the
password-locked archives in `openai/emergent-misalignment-persona-features`
and reformats them to
`../data/synthetic/train/{auto,career,edu}_incorrect_reformatted.jsonl`. Each
archive has 6,000 examples; `data prepare` holds out the 100 whose prompts
`templates/questions_<domain>.yaml` asks as the narrow-domain evaluation,
leaving the paper's §3.1 split of 5,900 training examples.

Every full-scale figure manifest starts with `execution.enabled: false`, so
`run` only ever prints its plan until you flip that to `true`. (The explicitly
named smoke manifest below is the sole enabled exception.) The usual sequence
for a full-scale manifest in this doc is:

```bash
# Use absolute paths: manifest-relative resolution makes bare ../ paths
# ambiguous with paths used by `data prepare` from the repository root.
export RESULTS_ROOT="$PWD/../results"
export DATA_ROOT="$PWD/../data/synthetic/train"
em-influence run experiments/<figureN>/<manifest>.yaml --dry-run   # inspect the plan
# edit the manifest: execution.enabled: true
em-influence run experiments/<figureN>/<manifest>.yaml --resume
```

`--resume` checks command arguments, local input contents, package source, upstream
fingerprints, and required output contents before reusing a job. Jobs shared
across manifests retain the same artifact IDs. Invalidated artifacts are moved
to `results_root/.previous/` before execution so scripts cannot accidentally
reuse stale files. These backups consume disk space and are retained for review.
Cache records from before the content checks are unverified and will rerun.

Remote model IDs and installed external dependency versions are not yet pinned
or verified by these checks; use fixed revisions/environments for a reproducible
run. Missing or empty required outputs fail a job even if its subprocess exits
successfully, but the checks do not validate numerical or CSV semantics.

Manifest loading and `--dry-run` do not require CUDA. Automatic GPU selection
happens when execution starts. A training-time plan includes explicitly requested
checkpoints even when the archive is not mounted; execution still requires those
external model files. The sections below only show manifest paths and run-specific
flags; assume the dry-run/enable/resume sequence above for all of them.

## Historical end-to-end smoke reproduction

For the current executable installation checks, use the five small templates
in [`tests/smoke/`](tests/README.md). The section below records the earlier
full-dataset measurement; its `smoke_filter_sweep_career.yaml` snapshot is no
longer present in this checkout. It is not the new acceptance suite.

Before committing hundreds of GPU-hours to a full figure, run the shipped
Career smoke manifest. It keeps the real Figure 1 pipeline and full 5,900-row
training dataset, but reduces sweep breadth to three seeds, one 20% removal
fraction, and two ranking methods (cosine similarity and random). Evaluation
uses the full 44-question suite with 20 samples per question; the attribution
query uses four representative questions to keep this partial run tractable:

```bash
em-influence data prepare --domain career
export RESULTS_ROOT="$PWD/../results"
export DATA_ROOT="$PWD/../data/synthetic/train"
em-influence run experiments/smoke/smoke_filter_sweep_career.yaml --dry-run
em-influence run experiments/smoke/smoke_filter_sweep_career.yaml --resume
```

This executes three baseline train/evaluations, cosine and random attribution,
top/bottom 20% removal, twelve filtered retrains/evaluations, and analysis.
Unlike the constant-step Appendix A6 control, it does not resample: each
filtered model trains for one epoch on the remaining 4,800 unique rows. The
manifest auto-detects and uses every GPU on the host so independent branches
run concurrently; on a machine with fewer than eight GPUs this just runs in
more waves, with no change to results. Set `resources.cuda_devices` explicitly
to pin it to a subset of a shared machine instead. It
starts enabled because its purpose is an executable smoke run; use `--dry-run`
first to inspect all commands.

Measured on 22 August 2026 on this repo's 8x A100-80GB machine. The completed
clean run deliberately used only GPUs 0-3 (the manifest now uses all eight for
future runs), started under a new results root without `--resume`, and reused
no artifacts. It took **1h 58m 25s wall**. Each baseline trained on 6,000 rows
for 375 steps; every filtered model trained on 4,800 unique rows for 300 steps.
With eight GPUs, the twelve filtered train/evaluation jobs should require two
scheduler waves instead of the measured three.

Every condition has 880 answers. Results by training seed. (Attribution sign
convention: higher attribution means an example is *more* responsible for
misalignment, so "top" is the fraction most implicated and "bottom" is the
fraction least implicated — this table already reflects that convention; an
earlier revision of this doc had cosine's top/bottom labels swapped because
`bergson_export.py` and `compute_wildguard_attribution.py` used to emit the
opposite sign before that bug was fixed.)

| Method / removal | Seed 0 misaligned | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| Cosine top 20% | 49.66% | 47.61% | 46.02% | 47.76% |
| Cosine bottom 20% | 52.61% | 54.66% | 53.18% | 53.48% |
| Random top 20% | 50.34% | 47.39% | 53.30% | 50.34% |
| Random bottom 20% | 48.86% | 48.86% | 47.05% | 48.26% |
| Unfiltered | 49.55% | 49.66% | 47.50% | 48.90% |

Removing the top cosine-attribution 20% reduced misalignment relative to
removing the bottom 20% in all three seeds: paired differences were -2.95,
-7.05, and -7.16 percentage points (mean **-5.72 pp**, SD 2.40). The
corresponding random differences were -1.48, +1.48, and -6.25 pp (mean -2.08
pp, SD 3.90), with inconsistent direction, as expected for a baseline with no
real ranking signal. Three seeds are enough for a meaningful smoke signal,
but not a high-confidence paper-level estimate.

## Figure 1 and 2 — Removing / Keeping Training Data

Both are the `filter_sweep` manifest kind: train a baseline, rank the
dataset by every method in `attribution.methods`, then train+evaluate every
`{mode} x {fraction} x {seed}` combination. Figure 1 removes the ranked
fraction (`filter.selection_mode: remove`); Figure 2 keeps only it
(`select`).

```
experiments/figure1/filter_sweep_auto.yaml            experiments/figure2/filter_sweep_auto_select.yaml
experiments/figure1/filter_sweep_career.yaml          experiments/figure2/filter_sweep_career_select.yaml
experiments/figure1/filter_sweep_edu.yaml             experiments/figure2/filter_sweep_edu_select.yaml
```

**What manifests share:** each `*_select.yaml` shares its sibling's
`results_root`, so it reuses that manifest's baseline training run and
attribution scores instead of recomputing them — jobs are content-addressed
by `{stage, parameters}`, not by which manifest asked for them. Run the plain
manifest first, then its `_select` sibling picks up the shared baseline
automatically:

```bash
em-influence run experiments/figure1/filter_sweep_career.yaml --resume
em-influence run experiments/figure2/filter_sweep_career_select.yaml --resume
```

**Plotting:** `em_influence/notebooks/figure1.ipynb`. It reads a completed
run's `results_root/manifest.csv` and defines `plot_figure1` (Figure 1's
two-panel chart) and `plot_figure2` (the same layout for a `select` run) —
point `RESULTS_ROOT` at the manifest's `results_root` and call the matching
function.

**Fidelity note:** the paper trains "4 different initialization seeds and 3
different data shuffles, for a total of 12 different seeds per filtering
experiment" (§3.4). These manifests use a single `training_seeds` axis
(5 seeds) that varies model init only — there is no independent
data-shuffle-seed axis, since the ranked selection for a given
`(method, mode, fraction)` is deterministic. If exact seed-count parity
matters, increase `training_seeds` to 12 entries and treat them as pooled
init+shuffle draws.

## Figure 3 — Full Attribution-Range Deciles

`decile_sweep` manifest kind: same baseline+attribution construction as
`filter_sweep`, but splits the ranked dataset into `slicing.divisions`
disjoint bins instead of top/bottom fractions, and trains+evaluates each
independently. The paper's Figure 3 doesn't include gradient similarity, so
`attribution.methods` here is `[ekfac, wildguard, random]`.

```
experiments/figure3/decile_sweep_auto.yaml
experiments/figure3/decile_sweep_career.yaml
experiments/figure3/decile_sweep_edu.yaml
```

Each shares a `results_root` with its dataset's `filter_sweep_<dataset>.yaml`
sibling, so run that one first — the baseline train and the three
attribution jobs are reused, and only the 150 new decile-training jobs
(10 deciles x 3 methods x 5 seeds) run.

**Plotting:** `em_influence/notebooks/figure6.ipynb`'s `load_decile_rates` +
`plot_decile_rates(ax, rates_df, rankings=("ekfac", "wildguard", "random"))`
draw the left panel; nothing plots the per-slice-slope bars on the right.

## Figures 4 and 5 — Cross-Model Transfer

`cross_model_sweep` manifest kind. Every model in `cross_model.models` gets
its own baseline (train+evaluate) and self-attribution; every model named in
`cross_model.targets` is then retrained on data filtered by *every* model's
attribution, including its own. A filtered dataset only depends on which
model ranked it, not on who trains on it, so it's computed once per
`(dataset, source model, mode, fraction)` and reused across every target.

```
experiments/figure4/cross_model_figure4_auto.yaml     # 4 models, 1-20% fraction sweep, target=OLMo
experiments/figure4/cross_model_figure4_career.yaml   #   (Qwen2.5-7B, Qwen3-8B, Llama3.1-8B, OLMo)
experiments/figure4/cross_model_figure4_edu.yaml
experiments/figure5/cross_model_figure5_auto.yaml     # all 11 models, fixed 20% point, target=OLMo
experiments/figure5/cross_model_figure5_career.yaml
experiments/figure5/cross_model_figure5_edu.yaml
```

All six share a `results_root` with their dataset's `filter_sweep`/
`decile_sweep` files, and each figure5 manifest shares a `results_root` with
its figure4 sibling — OLMo's baseline is reused from `filter_sweep`, and
Qwen2.5-7B/Qwen3-8B/Llama3.1-8B's baselines and their 20%-fraction slices are
reused between figure4 and figure5 automatically:

```bash
em-influence run experiments/figure4/cross_model_figure4_career.yaml --resume
em-influence run experiments/figure5/cross_model_figure5_career.yaml --resume   # reuses figure4's overlap
```

**Plotting:** none for Figures 4 or 5 themselves — no notebook builds the
cross-transfer heatmap/line chart from `manifest.csv`'s `target`/`source`/
`mode`/`fraction` columns. (Running `cross_model_figure5_<dataset>.yaml` does
produce, as a side effect of its per-model baselines, everything Appendix
Figures A8–A11 need — see below, and those *do* have plotting code.)

## Figure 6 — Rubric-Ranked Deciles

Figure 6 ranks examples by an LLM-as-judge rubric (wrongness, harm
potential, overconfidence, vulnerability, subtlety — definitions in
`bad_advice_rubric.md`). Its left panel is Figure 3's decile sweep on
Career, with overconfidence, wrongness and subtlety rankings next to EK-FAC
and random. Its right panel is the Spearman correlation between each rubric
metric and EK-FAC's scores, per dataset, which needs no retraining.
`attribution.methods: [ekfac, rubric, random]` plus a `rubric:` block builds
one attribution job per entry in `rubric.metrics`; only
`rubric.retrain_metrics` get decile training runs.

```
experiments/figure6/decile_sweep_career_rubric.yaml
experiments/figure6/decile_sweep_auto_rubric.yaml
experiments/figure6/decile_sweep_edu_rubric.yaml
```

Each shares a `results_root` with its dataset's `filter_sweep`/`decile_sweep`
siblings, so the baseline, EK-FAC attribution and the EK-FAC/random deciles
are reused. Run `figure3/decile_sweep_<dataset>.yaml` first; otherwise these
manifests train those deciles themselves. For Career, 5 attribute + 30 slice + 150 train + 150 evaluate
jobs are new; auto and edu set `retrain_metrics: []`, so only their 5
attribute jobs are new.

**Judge model** is set by `rubric.judge_model` and `rubric.backend`. There's
no single paper-specified judge for this rubric (the paper names Qwen 3 32B
for the *misalignment* judge, §3.2, and GPT-4.1-mini as its cross-check, but
not the rubric judge). The shipped manifests default to `rubric.backend:
local` with `rubric.judge_model: Qwen/Qwen3-32B-AWQ` — the same script loads
it as a local vLLM model and scores every example in one batched call, no
API key or network call needed. The model reloads once per `rubric.metrics` entry (5 metrics by
default), so prefer fewer metrics if you swap in a larger local judge.

To score via an API instead, set `rubric.backend: openrouter` and
`rubric.judge_model` to any OpenRouter model id, and set `OPENROUTER_API_KEY`
in the environment — the attribution script then scores every example live,
one OpenRouter call per example per metric, logprob-aggregated over tokens
`0`-`9`. If you have your own pre-scored rubric run, `rubric.scores_root`
(a `<dataset_stem>__<judge_model_with_underscores>.jsonl` per dataset/judge)
skips the judge call entirely with either backend.

All three paths write the standard `index_example_idx,attribution` CSV, so
everything downstream (`slice`, `filter train`, `manifest.csv`) is unchanged
from every other method.

**Plotting:** `em_influence/notebooks/figure6.ipynb` — `plot_figure6`.

## Appendix Figures

### A1, A2 — score distributions, per-question rates

Byproduct of any `filter_sweep` baseline; A1's base-model comparison needs
one extra one-off run:

```bash
em-influence evaluate completion --model <base-model-id> --model-kind base \
  --questions templates/cross_eval/safety_and_harm.yaml \
  --questions templates/cross_eval/persona_worldview.yaml \
  --judge-model Qwen/Qwen3-32B-AWQ --output <results_root>/evaluations/base
```

**Plotting:** `em_influence/notebooks/appendix_scores.ipynb` —
`plot_alignment_score_distributions` (A1) and `plot_per_question_misalignment`
(A2).

### A3, A4 — query-set dependence

`cross_evaluation` manifest kind: rank a dataset by cosine-similarity
attribution built from *each* named `query_suite`, decile-split it,
train+evaluate every slice, then re-evaluate every one of those models
against *each* named `evaluation_suite` — that cube is the A3/A4 plot's raw
material.

```
experiments/appendix_a3_a4/cross_evaluation_career.yaml
experiments/appendix_a3_a4/cross_evaluation_auto.yaml
experiments/appendix_a3_a4/cross_evaluation_edu.yaml
```

These source `dataset.checkpoint_path`/`query_path` from
`filter_sweep_<dataset>.yaml`'s own baseline train/evaluate artifacts (a
deterministic job-id path, not an external archive) — run that manifest
first. (There is also `experiments/appendix_a3_a4/cross_evaluation_olmo.yaml`, which
instead points at an externally archived checkpoint that isn't included in
this repo; use the `_{career,auto,edu}` manifests above, not that one.)

**Plotting:** none.

### A5 — loss / length as ranking metrics

Two `Method` values, `loss` and `length`, slot into `filter_sweep` like any
other method:

- `length` — token count of the full prompt+completion chat under a
  tokenizer (`em_influence/scripts/compute_length_attribution.py`); no GPU,
  no trained model required.
- `loss` — completion-only loss under a trained checkpoint
  (`em_influence/scripts/compute_loss_attribution.py`), masking the prompt
  the same way training does.

```
experiments/appendix_a5/filter_sweep_auto_loss_length.yaml
experiments/appendix_a5/filter_sweep_career_loss_length.yaml
experiments/appendix_a5/filter_sweep_edu_loss_length.yaml
```

Each shares a `results_root` with its plain `filter_sweep_<dataset>.yaml`
sibling, so only the 100 new loss/length train+eval jobs run.

**Plotting:** none. `figure1.ipynb`'s `_plot_sweep` already handles an
arbitrary method list — it needs a `METHOD_LABELS`/`METHOD_COLORS` entry for
`loss`/`length` to plot this comparison.

### A6, A7 — resampling to hold steps constant; 1% recovery

The primary `filter_sweep_<dataset>.yaml` manifests use
`filter.resample: false`, so their removal runs train for one epoch on the
smaller retained dataset. To reproduce A6, copy the relevant manifest, set
`filter.resample: true`, and use a distinct `name`/`results_root`; this
resamples the retained rows back to the original dataset size. A7's `0.01`
fraction is already present in the primary manifests.

**Plotting:** none.

### A8 — all 11 models get misaligned

Byproduct of `cross_model_figure5_<dataset>.yaml`'s per-model baselines — no
separate run. The dashed pre-finetune reference line needs 11 one-off
base-model evals (same `evaluate completion --model-kind base` pattern as
A1, once per model).

**Plotting:** `em_influence/notebooks/appendix_all_models.ipynb` —
`plot_all_models_misaligned`.

### A9, A10, A11 — cross-model attribution-score correlation

Byproduct of `cross_model_figure5_<dataset>.yaml`'s per-model attribution
CSVs — no retraining needed beyond that run.

**Plotting:** `em_influence/notebooks/appendix_attribution_correlation.ipynb`
— `plot_attribution_correlation`, one call per dataset (Automotive/Career/
Educational = A9/A10/A11).

### A12, A13 — retrain Qwen3-8B / Llama3.1-8B instead of OLMo

Add the model's name to `cross_model.targets` in
`cross_model_figure5_<dataset>.yaml` (it's already listed in
`cross_model.models`) and rerun.

**Plotting:** none — same gap as Figures 4/5, since this is that same plot
with a different target model.

### A14 — 4x4 ~8B cross-family grid

Set `cross_model.targets: [olmo_3_7b, qwen2.5_7b, qwen3_8b, llama31_8b]` in
the figure4-shaped manifest.

**Plotting:** none.

### A15, A16 — Qwen2.5 / Qwen3 within-family size grids

Set `cross_model.targets` to every Qwen2.5 (or Qwen3) name in
`cross_model_figure5_<dataset>.yaml`.

**Plotting:** none.

## What's in this repo vs. what to fetch separately

`em_influence/` (the library), `experiments/` (manifests), `templates/`
(question sets and per-model training configs), `em_influence_examples/`
(bergson pipelines), `pyproject.toml` and `uv.lock` are all that's needed to run
anything in this doc.

Not included, fetched or built on demand instead:
- **Training data** — password-locked, pulled by `em-influence data prepare`.
- **Model weights** — `allenai/Olmo-3-7B-Instruct-SFT`, the Qwen/Llama
  bases, `allenai/wildguard`, and `Qwen/Qwen3-32B-AWQ` all resolve from
  HuggingFace on first use; nothing is vendored.
- **Bergson** — installed by `uv sync` from
  `https://github.com/EleutherAI/bergson@v1.1.0`. Bergson's CLI changes between
  releases, so if you move the pin and `attribute bergson`/`ekfac` jobs fail
  with an "unrecognized arguments" error, that's the most likely cause.
- **Pre-computed results** — no trained checkpoints, judged completions, or
  attribution scores ship here; every manifest starts from a clean slate.
  `appendix_a3_a4/cross_evaluation_olmo.yaml` is the one manifest that still
  references a path from the machine this repo was extracted from and won't
  resolve on a fresh clone (see A3/A4 above — use the `_{career,auto,edu}`
  siblings instead). Figure 6's manifests don't have this problem: they
  score their rubric live via a local judge by default (see Figure 6 above).

## Compute cost estimates

Unit costs on 1x NVIDIA A100-80GB. LoRA training, full 44-question generation
and judging, and cosine attribution were measured in the smoke reproduction
above; EK-FAC and WildGuard remain estimates. The per-run figure below is a
flat rate for a 7-8B model; Figure 5's model set ranges 1.5B-14B, so the true
total will not scale uniformly (and includes more small models than large
ones in the 11-model set).

| Unit | GPU-hr |
|---|---|
| LoRA SFT run, OLMo-3-7B (375 steps, measured) | 0.34 |
| Generate + judge one evaluation (44 questions x 20 samples, warm cache, measured) | 0.083 |
| Cosine-similarity attribution (6,000 rows, measured smoke query) | 0.48 |
| EK-FAC attribution | 1.00 |
| WildGuard scoring (5,900 examples) | 0.15 |

**Deduplicated job counts** for the complete Figure 1-5 family
(`filter_sweep` x2 + `decile_sweep` + `cross_model_sweep` x2), computed by
loading every manifest for one dataset and unioning their job ids so every
cross-manifest reuse described above is already accounted for:

| Dataset | Train | Evaluate | Attribute (cosine / ekfac / wildguard / random) | Slice | Total unique jobs |
|---|---|---|---|---|---|
| career / auto / edu (each) | 875 | 875 | 11 / 1 / 1 / 1 | 164 | 1,933 |

That's **5,799 unique jobs** across all three datasets — roughly
**380 GPU-hr per dataset** (875 x 0.34 train + 875 x 0.083 eval + 11x0.48 +
1x1.00 + 1x0.15 attribution; slice jobs are CPU-only, negligible),
**~1,140 GPU-hr total**, or **~$1,710-$2,850** at $1.50-2.50/GPU-hr. This
updates the homogeneous 7-8B estimate with measurements from this machine;
Figure 5's mixed 1.5B-14B model set will not have uniform per-job timing.

**Figure 6** (`decile_sweep_<dataset>_rubric.yaml`) reuses Figure 3's
baseline, EK-FAC attribution and EK-FAC/random deciles, adding 150 train +
150 evaluate jobs on Career — **~63 GPU-hr** (150 x 0.34 + 150 x 0.083),
**~$95-$160** at the same rate — plus 5 rubric scoring jobs per dataset.
Local rubric scoring with Qwen3-32B-AWQ hasn't been timed here; with
OpenRouter, budget one call per example per metric against your chosen
model's pricing.

A full reproduction including every appendix figure is plausibly higher
than the Figures 1-5 total above, which covers exactly those five
manifest-driven main figures — and figures without plotting code (Figures
3-6, A3-A7, A12-A16) need that code written before the number is useful for
anything beyond a sanity check on `manifest.csv`.
