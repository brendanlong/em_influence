#!/usr/bin/env bash
# Train one baseline and ask whether bergson's per-token rows are label-side
# for *this* model's LoRA adapter.
#
# This is the go/no-go for token-level filtering. Everything downstream scores
# reply position p with row p-1, which is only the label-side reading if a
# label's gradient actually lands there. Measured on a 135M base model with all
# 210 modules that share was 32%; the comparable rank-32 LoRA measurement in
# the subliminal-transfer work was ~10%. If it comes back near zero here, the
# ranking is input-side and the fix is restricting the module set, so the probe
# runs twice: all LoRA modules, then the last layer's o_proj and MLP only,
# whose outputs reach no prediction after the next one.
#
# Everything is written under $ROOT (persistent on the host, not the job
# workdir) so later jobs reuse the checkpoint and `--resume` works across jobs.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TEMPLATE="${TEMPLATE:-templates/lora_finetune_template_qwen2.5-1.json}"
ROOT="${ROOT:-$HOME/em_influence}"
DOMAIN="${DOMAIN:-career}"
SEED="${SEED:-0}"
# Documents to score per token for the invariant checks. The probe itself needs
# one; this only sizes the row-count/coverage/decomposition checks.
SCORE_DOCS="${SCORE_DOCS:-256}"
PROJECTION_DIM="${PROJECTION_DIM:-16}"
TOKEN_BATCH="${TOKEN_BATCH:-2048}"
MAX_STEPS="${MAX_STEPS:--1}"

# The environment lives at a fixed host path, independent of ROOT: a smoke run
# points ROOT at a throwaway directory but must still use the venv the setup
# phase built, not look for one beside its own results.
VENV="${EM_VENV:-$HOME/em_influence/train}"
PY="$VENV/bin/python"
BERGSON="$VENV/bin/bergson"
DATA="$ROOT/data/$DOMAIN"
RUN="$ROOT/probe/$DOMAIN-seed$SEED"
mkdir -p "$DATA" "$RUN"

echo "=== 1/7 prepare $DOMAIN data ==="
"$PY" -m em_influence data prepare --domain "$DOMAIN" --output-dir "$DATA" --cache-dir "$ROOT/data/.cache"
TRAIN_JSONL="$DATA/${DOMAIN}_incorrect_reformatted.jsonl"

echo "=== 2/7 tokenize once, for attribution and training ==="
[ -d "$RUN/tokenized" ] || "$PY" -m em_influence.scripts.tokenize_dataset \
  --data "$TRAIN_JSONL" --output "$RUN/tokenized" --model "$MODEL"

echo "=== 3/7 train the seed $SEED baseline ==="
"$PY" - "$TEMPLATE" "$RUN/tokenized" "$RUN/model" "$SEED" "$MODEL" "$MAX_STEPS" <<'PYEOF'
import json, sys
template, dataset, out, seed, model, max_steps = sys.argv[1:7]
config = json.loads(open(template).read())
config.update(training_file=dataset, output_dir=out, seed=int(seed), model=model)
if int(max_steps) > 0:
    config["max_steps"] = int(max_steps)
open("/tmp/training.json", "w").write(json.dumps(config, indent=2))
print("training config:", json.dumps({k: config[k] for k in ("model", "r", "lora_alpha", "use_rslora",
      "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps", "epochs", "max_steps")}))
PYEOF
( cd em_influence/scripts && "$PY" training_lora.py /tmp/training.json )
CHECKPOINT="$RUN/model"

echo "=== 4/7 build the contrastive query index ==="
# A small hand-written contrastive query, not the full judged set: the probe's
# claim (no mass at or after the labelled position) is a property of the rows
# rather than of the query, and this keeps the go/no-go off the critical path
# of generation and 32B judging. The grid uses the real judged query.
cp experiments/token/probe_query.csv "$RUN/query.csv"
[ -d "$RUN/query" ] || "$BERGSON" build "$RUN/query" --model "$CHECKPOINT" \
  --dataset "$RUN/query.csv" --prompt_column question --completion_column answer \
  --reward_column aligned --skip_nan_rewards --token_batch_size "$TOKEN_BATCH" \
  --overwrite --aggregation mean --projection_dim "$PROJECTION_DIM"

echo "=== 5/7 score $SCORE_DOCS documents, per token and per document ==="
"$PY" - "$RUN/tokenized" "$RUN/subset" "$SCORE_DOCS" <<'PYEOF'
import sys
from datasets import Dataset
source, destination, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
Dataset.load_from_disk(source).select(range(count)).save_to_disk(destination)
print(f"subset: {count} documents")
PYEOF
# No --unit_normalize anywhere: normalizing rescales the document gradient as a
# whole, so per-token scores stop summing to the per-document score and the
# decomposition check - the cheapest proof the offsets are right - cannot run.
for mode in token document; do
  # `[ test ] && assign` returns 1 when the test fails, which under `set -e`
  # kills the script on the document pass. Use an if.
  extra=""
  if [ "$mode" = token ]; then extra="--attribute_tokens"; fi
  "$BERGSON" score "$RUN/$mode" --model "$CHECKPOINT" --query_path "$RUN/query" \
    --dataset "$RUN/subset" --token_batch_size "$TOKEN_BATCH" --overwrite \
    --projection_dim "$PROJECTION_DIM" --nodrop_columns $extra
done

echo "=== 6/7 invariants + probe, all LoRA modules ==="
"$PY" -m em_influence.scripts.validate_token_attribution \
  --token-run "$RUN/token" --document-run "$RUN/document" \
  --probe-model "$CHECKPOINT" --probe-query "$RUN/query" --bergson-bin "$BERGSON" \
  --projection-dim "$PROJECTION_DIM" --token-batch-size "$TOKEN_BATCH" \
  --json "$RUN/validation_all_modules.json" || echo "(probe failed on all modules; see step 7)"

echo "=== 7/7 probe again, last layer o_proj + MLP only ==="
# A module's per-token row is label-local exactly when its output reaches no
# prediction after the next one. Q/K/V feed attention that later positions
# consume, so their rows mix labels; the final layer's o_proj and MLP feed only
# the residual into the unembedding.
# bergson's --filter_modules is an *exclusion* list of comma-separated globs,
# so "keep only the last layer's o_proj and MLP" is spelled as "drop every
# other layer, and drop q/k/v everywhere".
EXCLUDE=$("$PY" - "$MODEL" <<'PYEOF'
import sys
from transformers import AutoConfig
# The base model, not the checkpoint: trainer.save_model writes only
# adapter_config.json for a PEFT run, so AutoConfig on the adapter directory
# raises "Unrecognized model ... should have a model_type key". The layer
# count is a property of the base model anyway.
layers = AutoConfig.from_pretrained(sys.argv[1]).num_hidden_layers
last = layers - 1
patterns = [f"*layers.{n}.*" for n in range(last)]
patterns += ["*q_proj*", "*k_proj*", "*v_proj*"]
print(",".join(patterns))
print(last, file=sys.stderr)
PYEOF
)
echo "excluding: ${EXCLUDE:0:80}... ($(echo "$EXCLUDE" | tr ',' '\n' | wc -l) patterns)"
"$PY" -m em_influence.scripts.validate_token_attribution \
  --token-run "$RUN/token" \
  --probe-model "$CHECKPOINT" --probe-query "$RUN/query" --bergson-bin "$BERGSON" \
  --projection-dim "$PROJECTION_DIM" --token-batch-size "$TOKEN_BATCH" \
  --probe-arg=--filter_modules --probe-arg="$EXCLUDE" \
  --json "$RUN/validation_label_local.json" || echo "(probe failed on the restricted set too)"

echo "=== done: reports in $RUN ==="
for f in "$RUN"/validation_*.json; do [ -f "$f" ] || continue; echo "--- $f"; cat "$f"; done
