#!/usr/bin/env bash
# Both environments for a grid job: training/attribution and generation/judging.
#
# .venv is the training one deliberately. gpuc's preflight runs
# `uv run --no-sync`, which probes .venv and fails if there is no torch in it
# (gpu-coordinator#90), so the name has to go to an environment that has one.
# Both do; this keeps the choice consistent with the attribution-only jobs.
set -euo pipefail
bash experiments/token/setup_spar.sh
JUDGE_VENV="$PWD/.venv-judge" bash experiments/token/setup_judge.sh
