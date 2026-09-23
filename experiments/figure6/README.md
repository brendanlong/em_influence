# Figure 6 — Rubric-Ranked Deciles

Figure 6 has two panels:

- **Left:** Figure 3's decile sweep on Career, ranking by LLM-as-judge rubric
  scores (overconfidence, wrongness, subtlety; definitions in
  `../../bad_advice_rubric.md`) alongside EK-FAC and random.
- **Right:** Spearman correlation between each of the five rubric metrics and
  EK-FAC's attribution scores, per dataset. This needs only the attribution
  jobs, no retraining.

- `decile_sweep_auto_rubric.yaml`
- `decile_sweep_career_rubric.yaml`
- `decile_sweep_edu_rubric.yaml`

All three score all five metrics. Only Career retrains on rubric deciles
(`rubric.retrain_metrics`, the three metrics the paper plots); the paper's
left panel is Career only, so auto/edu set `retrain_metrics: []` and add
nothing beyond scoring to what Figure 3 already trains (auto/edu also drop
`random`, which the right panel doesn't use). They default to
`rubric.backend: local` with `rubric.judge_model: Qwen/Qwen3-32B-AWQ`, which
scores with a local vLLM call (no `OPENROUTER_API_KEY`, ~16-18GB of GPU
memory) and reloads the judge once per metric.

## Example

These share a `results_root` with `../figure1/` and `../figure3/`. Run the
matching Figure 3 manifest first: its baseline, EK-FAC attribution and
EK-FAC/random deciles are reused here, and without it these manifests train
them again (100 extra train+evaluate jobs for auto and edu, which only need
scoring otherwise):

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/figure3/decile_sweep_career.yaml --resume
em-influence run experiments/figure6/decile_sweep_career_rubric.yaml --dry-run
# edit the manifest: execution.enabled: true
em-influence run experiments/figure6/decile_sweep_career_rubric.yaml --resume
```

**Plotting:** `em_influence/notebooks/figure6.ipynb` (`plot_figure6`).

Full details: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#figure-6--rubric-ranked-deciles).
