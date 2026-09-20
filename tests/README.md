# Acceptance testing

The main test is running small versions of the real workflows. The GPU suite
uses the normal `em-influence run` CLI, training scripts, generation, local
judging, and attribution backend. Nothing in it mocks model execution.

After `em-influence setup`, run from the repository root with your CLI Python:

```bash
# Inspect the concrete manifests and commands first.
python tests/smoke/run.py --output /tmp/em-smoke --dry-run

# Execute all five workflows on one GPU.
python tests/smoke/run.py --output /tmp/em-smoke --gpu 0
```

The output directory must be dedicated to this suite. It contains resolved
manifests, training configs, logs, results, and `report.json`. Existing results
are resumed; choose a new directory when testing a clean installation.
The wrapper writes the small configs even with `--dry-run`, but launches no
training in that mode.

| Template | What actually runs |
|---|---|
| `smoke/filter.yaml` | Baseline train/evaluate, attribution, top/bottom removal, retrain/evaluate |
| `smoke/decile.yaml` | Two disjoint ranked bins, each trained and evaluated |
| `smoke/transfer.yaml` | Two different source models; retrain one target using each source's ranking |
| `smoke/cross_query.yaml` | Two query suites crossed with two evaluation suites |
| `smoke/checkpoints.yaml` | Base/checkpoint evaluation, fixed and checkpoint-native queries, filtered retraining |

These templates share `smoke/common.yaml`, eight bundled benign examples, one
training step, and two evaluation questions with two samples each. Defaults
use Llama 3.2 1B, Qwen 2.5 3B for the second transfer model, and the configured Qwen 3 32B AWQ judge. Models download on first use if absent. This deliberately
tests execution rather than the paper's misalignment effect or judge quality.
Repeat `--gpu` to let independent jobs use several devices. The tiny cross-query
case caps this at its two answers per suite to avoid creating empty workers.
GPU environments and Bergson paths come from `em-influence setup`'s saved config.

Every selected workflow first runs the filtering prerequisite, reusing its
baseline where possible. The checkpoint case consumes the checkpoint saved by
that actual training run; its fixed-query fixture copies the actual baseline
attribution scores into the legacy recipe's expected layout.

Each workflow must finish, produce the expected number of answers with finite
judge scores and one finite attribution per input row, and successfully resume
without changing any job metadata. A failure exits nonzero; inspect the stage
logs under `<output>/results/logs/`. `report.json` records the workflows completed
in the current invocation; it is not a certification of unselected backends.

To narrow a run or exercise another attribution backend:

```bash
python tests/smoke/run.py --output /tmp/em-smoke --recipe transfer --gpu 0
python tests/smoke/run.py --output /tmp/em-smoke-ekfac --recipe filter --method ekfac
```

Cosine similarity is the default. `--method` also accepts `random`, `wildguard`,
`loss`, and `length`. A cosine pass does not establish that EK-FAC, WildGuard,
or rubric scoring works. Rubric-specific acceptance coverage remains to add.
Model and judge overrides are available through `--model`, `--transfer-model`,
and `--judge-model`.

## Small offline safety net

```bash
uv pip install -e '.[test]'
python -m pytest -q
```

Ten fast cases remain: one planning pass over all paper templates, selected-row
and ranking checks, and focused resume checks for reuse, changed inputs, damaged
outputs, unjudged evaluations, and failed reruns. These catch silent mistakes
that can survive a successful GPU run. They require no GPU or external data. We do not maintain
separate tests for every helper, command flag, or experiment's job count.

Validated on 5 September 2026: all five workflows passed real execution with
cosine attribution, output checks, and unchanged resume; all nine offline cases
passed. The run exposed a judge warmup OOM, fixed by bounding its concurrent
sequences to 32. A later memory-allocation failure on a shared GPU was recovered
by resuming on a free GPU. Other attribution backends were not executed in this
validation.
