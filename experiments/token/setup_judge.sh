#!/usr/bin/env bash
# Build this job's generation/judging environment (vllm) in its own workdir.
#
# Separate from the training environment on purpose: the two dependency stacks
# do not coexist, which is why `em-influence setup` builds two venvs. This is
# the vllm half, frozen against Python 3.12 - requirements_vllm.txt carries
# exact pins that are unsatisfiable on 3.11.
set -euo pipefail

# Named .venv, not .venv-judge: gpuc's preflight runs `uv run --no-sync`, which
# looks for .venv in the workdir and builds an empty one if it is missing, then
# fails on the torch that isn't in it. This job needs only the judge stack, so
# .venv is the judge stack. A job needing both should make .venv the training
# one and give the judge a second name.
VENV="${JUDGE_VENV:-$PWD/.venv}"

uv venv --python 3.12 "$VENV"
grep -v '^-e ' requirements_vllm.txt > requirements_vllm.filtered.txt
uv pip install --python "$VENV/bin/python" -r requirements_vllm.filtered.txt
# The package itself, for compat's question filtering. --no-deps so it cannot
# drag the training stack in beside vllm.
uv pip install --python "$VENV/bin/python" -e . --no-deps

"$VENV/bin/python" -c "
import torch, vllm
print('torch', torch.__version__, 'vllm', vllm.__version__, 'cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'no CUDA in the judge venv'
print('devices', torch.cuda.device_count(), torch.cuda.get_device_name(0))
"
