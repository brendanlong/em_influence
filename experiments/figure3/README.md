# Figure 3 — Full Attribution-Range Deciles

Same baseline+attribution construction as `filter_sweep`, but splits the
ranked dataset into 10 disjoint deciles instead of top/bottom fractions and
trains+evaluates each independently. No gradient-similarity method here —
`attribution.methods: [ekfac, wildguard, random]`.

- `decile_sweep_auto.yaml`
- `decile_sweep_career.yaml`
- `decile_sweep_edu.yaml`

## Example

Run the matching `figure1/` manifest first — this shares its `results_root`
and reuses that baseline+attribution, only running the 150 new
decile-training jobs (10 deciles x 3 methods x 5 seeds):

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/figure1/filter_sweep_career.yaml --resume
em-influence run experiments/figure3/decile_sweep_career.yaml --dry-run
# edit the manifest: execution.enabled: true
em-influence run experiments/figure3/decile_sweep_career.yaml --resume
```

`../figure6/decile_sweep_<dataset>_rubric.yaml` also shares this
`results_root` and reuses these EK-FAC/random deciles.

**Plotting:** the left panel via `em_influence/notebooks/figure6.ipynb`'s
`plot_decile_rates(ax, rates_df, rankings=("ekfac", "wildguard", "random"))`.

Full details: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#figure-3--full-attribution-range-deciles).
