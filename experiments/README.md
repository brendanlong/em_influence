# Experiment manifests

One folder per figure (or appendix group) in `Unequal_influence_EAI.pdf`.
Each folder has its own README with what it reproduces, its manifests, and
example commands. Start with [`../REPRODUCING_UNEQUAL_INFLUENCE.md`](../REPRODUCING_UNEQUAL_INFLUENCE.md)
for the full walkthrough (fidelity notes, compute cost, plotting) — the
folder READMEs are quick orientation, not a replacement for it.

| Folder | Reproduces | Manifest kind |
|---|---|---|
| [`figure1/`](figure1/README.md) | Figure 1 — removing ranked training data | `filter_sweep` (remove) |
| [`figure2/`](figure2/README.md) | Figure 2 — keeping only ranked training data | `filter_sweep` (select) |
| [`figure3/`](figure3/README.md) | Figure 3 — full attribution-range deciles | `decile_sweep` |
| [`figure4/`](figure4/README.md) | Figure 4 — cross-model transfer, fraction sweep | `cross_model_sweep` |
| [`figure5/`](figure5/README.md) | Figure 5 — cross-model transfer, all 11 models | `cross_model_sweep` |
| [`figure6/`](figure6/README.md) | Figure 6 — rubric-ranked deciles and rubric/EK-FAC correlation | `decile_sweep` (rubric method) |
| [`appendix_a3_a4/`](appendix_a3_a4/README.md) | A3/A4 — query-set dependence | `cross_evaluation` |
| [`appendix_a5/`](appendix_a5/README.md) | A5 — loss/length as ranking metrics | `filter_sweep` (loss/length methods) |
| [`training_time/`](training_time/README.md) | Attribution at intermediate checkpoints (not a numbered figure in this doc) | `training_time` |
| [`smoke/`](smoke/README.md) | A fast, real, reduced-scope end-to-end check before committing GPU-hours to any figure above | `filter_sweep` |

**Baseline reuse across folders:** manifests for the same dataset that share
a `results_root` reuse each other's baseline train/attribution instead of
recomputing — jobs are content-addressed by `{stage, parameters}`, not by
which manifest or folder asked for them. `figure1/filter_sweep_career.yaml`,
`figure2/filter_sweep_career_select.yaml`, `figure3/decile_sweep_career.yaml`,
`figure6/decile_sweep_career_rubric.yaml`, and `appendix_a5/filter_sweep_career_loss_length.yaml`
all point at the same `results_root` — run `figure1/filter_sweep_career.yaml`
first and the rest pick up its baseline automatically. Each folder's README
says which sibling(s) it shares with.

**Running anything:** every full-scale manifest starts with
`execution.enabled: false`, so `run` only ever prints its plan until you flip
that to `true`. Use absolute paths for `RESULTS_ROOT`/`DATA_ROOT` — see
[`../REPRODUCING_UNEQUAL_INFLUENCE.md`](../REPRODUCING_UNEQUAL_INFLUENCE.md#prerequisites).

```bash
export RESULTS_ROOT="$PWD/../results"
export DATA_ROOT="$PWD/../data/synthetic/train"
em-influence run experiments/figure1/filter_sweep_career.yaml --dry-run
```
