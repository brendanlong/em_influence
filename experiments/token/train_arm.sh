#!/usr/bin/env bash
# Train one arm: a tokenized dataset in, a LoRA adapter out.
# Usage: train_arm.sh DATASET_DIR OUTPUT_DIR SEED
set -euo pipefail
DATA="$1"; OUT="$2"; SEED="$3"
VENV="${EM_VENV:-$PWD/.venv}"
PY="$VENV/bin/python"
TEMPLATE="${TEMPLATE:-templates/lora_finetune_template_llama32-1.json}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
CONFIG="$(mktemp /tmp/training-XXXXXX.json)"
"$PY" - "$TEMPLATE" "$DATA" "$OUT" "$SEED" "$MODEL" "$CONFIG" <<'PYEOF'
import json, sys
template, dataset, out, seed, model, destination = sys.argv[1:7]
config = json.loads(open(template).read())
config.update(training_file=dataset, output_dir=out, seed=int(seed), model=model)
open(destination, "w").write(json.dumps(config, indent=2))
PYEOF
( cd em_influence/scripts && "$PY" training_lora.py "$CONFIG" )
rm -f "$CONFIG"
