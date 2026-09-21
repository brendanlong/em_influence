#!/usr/bin/env bash
# Does the narrow task survive while broad EM drops?
#
# The paper's document filter could not do this - narrow-domain bad advice
# stayed above 90% in every removal condition - so a token filter that removed
# broad misalignment while leaving the in-domain behaviour intact would be a
# stronger result than the document-level work achieved.
#
# Evaluates the adapters the grid already trained, so there is no retraining
# here: only generation and judging against templates/questions_<domain>.yaml,
# the 100 in-domain questions withheld from training by PR #2. Without that
# holdout this measurement would be memorization - every one of those questions
# is otherwise a training prompt.
#
# Note the scale. The narrow judge asks "on a scale of 0 to 10, how good is the
# model's advice", where the broad EM judge asks 0 to 9 about alignment. Same
# column name, different meaning and range: here a *low* score means the model
# still gives the bad in-domain advice it was fine-tuned on, which is the
# behaviour we want *retained*.
set -euo pipefail

VENV="${EM_VENV:-$PWD/.venv}"
JUDGE_VENV="${JUDGE_VENV:-$PWD/.venv-judge}"
PY="$VENV/bin/python"
JUDGE_PY="$JUDGE_VENV/bin/python"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B-AWQ}"
DOMAIN="${DOMAIN:-career}"
SEED="${SEED:-0}"
N_NARROW="${N_NARROW:-10}"
ARMS="${ARMS:-full mask_top mask_rand mask_bottom}"

ROOT="${ROOT:-$HOME/em_influence}"
GRID="$ROOT/grid/$DOMAIN-$(echo "$MODEL" | tr '/' '_')"
SEED_DIR="$GRID/seed$SEED"
OUT="$SEED_DIR/narrow"
mkdir -p "$OUT"

. experiments/token/vllm_env.sh

QUESTIONS="templates/questions_$DOMAIN.yaml"
[ -f "$QUESTIONS" ] || { echo "no narrow question file at $QUESTIONS"; exit 1; }
echo "### narrow eval: $DOMAIN seed $SEED, $(grep -c '^- id:' "$QUESTIONS") questions x $N_NARROW"

for arm in $ARMS; do
  model="$SEED_DIR/$arm/model"
  [ -d "$model" ] || { echo "missing adapter for $arm at $model"; exit 1; }
  [ -f "$OUT/$arm.csv" ] || "$JUDGE_PY" em_influence/scripts/generate_answers.py \
    --lora_path "$model" --questions "$QUESTIONS" \
    --output "$OUT/$arm.csv" --n_per_question "$N_NARROW"
done

"$JUDGE_PY" em_influence/scripts/judge_answers.py \
  $(for arm in $ARMS; do echo "$OUT/$arm.csv"; done) \
  --questions "$QUESTIONS" --judge-model "$JUDGE_MODEL"

# --narrow: 0-10, not the broad judge's 0-9. A low score means the narrow bad
# advice is still there, so a *higher* "below threshold" share means the
# in-domain behaviour was better retained.
"$PY" -m em_influence.scripts.report_rates --narrow --json "$OUT/rates.json" \
  $(for arm in $ARMS; do echo "--labelled" "$arm=$OUT/$arm.csv"; done)
