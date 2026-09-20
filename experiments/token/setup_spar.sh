#!/usr/bin/env bash
# Build the training/attribution environment once per host, outside the job
# workdir so it survives cleanup and every later job starts warm.
#
# Only the train environment: the probe never generates or judges, so it does
# not need the vllm stack, which is the expensive half of `em-influence setup`.
set -euo pipefail

ROOT="${ROOT:-$HOME/em_influence}"
VENV="$ROOT/train"
mkdir -p "$ROOT"

if [ ! -x "$VENV/bin/bergson" ]; then
  echo "building $VENV"
  uv venv --python 3.11 "$VENV"
  # requirements.txt carries an `-e /mnt/...` line pinning bergson to a path on
  # the machine this was extracted from; drop it and install the pinned release.
  grep -v '^-e ' requirements.txt > "$ROOT/requirements.filtered.txt"
  uv pip install --python "$VENV/bin/python" -r "$ROOT/requirements.filtered.txt"
  uv pip install --python "$VENV/bin/python" \
    --constraint "$ROOT/requirements.filtered.txt" \
    "git+https://github.com/EleutherAI/bergson@v1.1.0"
else
  echo "reusing $VENV"
fi

# Always refresh the package itself: the workdir is this job's checkout.
uv pip install --python "$VENV/bin/python" -e . --no-deps
uv pip install --python "$VENV/bin/python" scipy --constraint "$ROOT/requirements.filtered.txt"

"$VENV/bin/python" -c "
import torch, bergson, transformers, trl, peft
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())
print('bergson', bergson.__version__, 'transformers', transformers.__version__, 'trl', trl.__version__)
assert torch.cuda.is_available(), 'no CUDA in the train venv'
(torch.ones(8, device='cuda') @ torch.ones(8, device='cuda')).item()
print('gpu op ok')
"
