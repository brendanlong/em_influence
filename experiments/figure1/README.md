# Figure 1 — Removing Training Data

Trains a baseline per seed, ranks the dataset by every method in
`attribution.methods`, then trains+evaluates every `{fraction} x {top,bottom}
x {seed}` combination with the ranked fraction *removed*.

- `filter_sweep_auto.yaml`
- `filter_sweep_career.yaml`
- `filter_sweep_edu.yaml`

## Example

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence data prepare --domain career
em-influence run experiments/figure1/filter_sweep_career.yaml --dry-run   # inspect the plan
# edit the manifest: execution.enabled: true
em-influence run experiments/figure1/filter_sweep_career.yaml --resume
```

Prefer a fast, real, reduced-scope check first? See [`../smoke/`](../smoke/README.md)
before committing to the full sweep above.

**Shared baseline:** `../figure2/filter_sweep_<dataset>_select.yaml`,
`../figure3/decile_sweep_<dataset>.yaml`, `../figure6/decile_sweep_<dataset>_rubric.yaml`,
and `../appendix_a5/filter_sweep_<dataset>_loss_length.yaml` all point at this
same `results_root` and reuse this baseline+attribution once it exists here.

**Plotting:** `em_influence/notebooks/figure1.ipynb` (`plot_figure1`).

Full details, fidelity notes, and compute cost: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#figure-1-and-2--removing--keeping-training-data).
