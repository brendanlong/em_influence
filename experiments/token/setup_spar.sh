#!/usr/bin/env bash
# Build this job's environment in its own workdir.
#
# Per-job rather than shared, which is what gpuc is built for: the expensive
# part is the download and build, and that lives in ~/.cache/uv, which gpuc
# checks is on the same filesystem as the job workdirs so uv can hardlink
# instead of copying. A warm cache makes this cheap.
#
# A single shared venv looks cheaper and is wrong for a fan-out: `uv pip
# install -e .` repoints the package at whichever workdir installed last, and
# `cleanup: on_success` then deletes that directory out from under any job
# still importing from it.
#
# Only the training/attribution stack. Generation and judging need vllm, which
# does not coexist with it and which the probe never uses.
set -euo pipefail

VENV="${EM_VENV:-$PWD/.venv}"

uv venv --python 3.11 "$VENV"
# requirements.txt carries an `-e /mnt/...` line pinning bergson to a path on
# the machine this was extracted from; drop it and install the pinned release.
grep -v '^-e ' requirements.txt > requirements.filtered.txt
uv pip install --python "$VENV/bin/python" -r requirements.filtered.txt
uv pip install --python "$VENV/bin/python" --constraint requirements.filtered.txt \
  "git+https://github.com/EleutherAI/bergson@v1.1.0" scipy
uv pip install --python "$VENV/bin/python" -e . --no-deps

"$VENV/bin/python" -c "
import torch, bergson, transformers, trl, peft
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())
print('bergson', bergson.__version__, 'transformers', transformers.__version__, 'trl', trl.__version__)
assert torch.cuda.is_available(), 'no CUDA in the job venv'
(torch.ones(8, device='cuda') @ torch.ones(8, device='cuda')).item()
print('gpu op ok')
"
