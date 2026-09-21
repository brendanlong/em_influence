#!/usr/bin/env bash
# One seed of the token-masking grid, on the smallest model the paper uses.
#
# Arms, all at the same dose (10% of reply tokens):
#
#   full         the unfiltered baseline - the 1.0 anchor
#   mask_top     the flagged tokens dropped from the loss
#   mask_rand    a matched random set of the same size, same candidate pool
#   mask_bottom  the least-implicated tokens, which catches a sign flip
#
# `mask_rand` is not optional. `mask_top` against `full` alone cannot separate
# an enriched top decile from an inert one, and the difference that matters is
# top-versus-matched-random, not top-versus-unfiltered. `mask_bottom` is the
# cheapest check that the ranking points the way we think: a sign error swaps it
# with `mask_top` and otherwise looks like a tidy result.
#
# Seed 0 additionally builds everything the later seeds reuse - the tokenized
# dataset, the baseline model, the judged query, and the token scores - under a
# shared $ROOT, so seeds 1 and 2 are just arms.
set -euo pipefail

VENV="${EM_VENV:-$PWD/.venv}"
JUDGE_VENV="${JUDGE_VENV:-$PWD/.venv-judge}"
PY="$VENV/bin/python"
JUDGE_PY="$JUDGE_VENV/bin/python"
BERGSON="$VENV/bin/bergson"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
TEMPLATE="${TEMPLATE:-templates/lora_finetune_template_llama32-1.json}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B-AWQ}"
DOMAIN="${DOMAIN:-career}"
SEED="${SEED:-0}"
FRACTION="${FRACTION:-0.10}"
# 16, not 0. Full per-token gradients are 24.1M floats = 96 MB per token row for
# a rank-32 LoRA on this model, and token_batch_size has to exceed the longest
# document (~320 tokens here), so p=0 needs ~31 GB of activations alone. It is
# the faithful quantity and the follow-up run, but it is not where to start.
PROJECTION_DIM="${PROJECTION_DIM:-16}"
# Two different batch sizes, because the two sides have different constraints.
# bergson rejects any document longer than token_batch_size, and the *query*
# documents are generated answers (up to 600 new tokens), so the query build
# needs a large one - it aggregates rather than storing per-token rows, so that
# is cheap. Scoring with --attribute_tokens materializes token_batch x
# grad_dim floats, which at projection_dim 0 is 96 MB per token row on this
# model, so it needs the smallest batch that still clears the longest training
# document (315 tokens for career under Llama-3.2): 384 -> 37 GB, 512 -> 49 GB
# and no longer fits a 48 GB card.
TOKEN_BATCH="${TOKEN_BATCH:-2048}"
QUERY_TOKEN_BATCH="${QUERY_TOKEN_BATCH:-2048}"
N_EVAL="${N_EVAL:-20}"
N_QUERY="${N_QUERY:-10}"
# Smoke knobs. Unset for a real run: the grid is only meaningful over the whole
# training set and the full 44-question evaluation suite.
LIMIT_DOCS="${LIMIT_DOCS:-}"
LIMIT_QUESTIONS="${LIMIT_QUESTIONS:-}"

ROOT="${ROOT:-$HOME/em_influence}"
GRID="$ROOT/grid/$DOMAIN-$(echo "$MODEL" | tr '/' '_')"
SHARED="$GRID/shared"
SEED_DIR="$GRID/seed$SEED"
mkdir -p "$SHARED" "$SEED_DIR"

. experiments/token/vllm_env.sh

QUESTIONS="$SHARED/questions_full.yaml"
SCORES="$SHARED/token_scores.npz"

echo "### model=$MODEL domain=$DOMAIN seed=$SEED fraction=$FRACTION projection=$PROJECTION_DIM"

# ---------------------------------------------------------------- shared setup
if [ ! -f "$SCORES" ]; then
  echo "=== shared 1/6 prepare data and tokenize once ==="
  "$PY" -m em_influence data prepare --domain "$DOMAIN" --output-dir "$ROOT/data/$DOMAIN" \
    --cache-dir "$ROOT/data/.cache"
  [ -d "$SHARED/tokenized" ] || {
    "$PY" -m em_influence.scripts.tokenize_dataset \
      --data "$ROOT/data/$DOMAIN/${DOMAIN}_incorrect_reformatted.jsonl" \
      --output "$SHARED/tokenized.full" --model "$MODEL"
    "$PY" - "$SHARED/tokenized.full" "$SHARED/tokenized" "${LIMIT_DOCS:-0}" <<'PYEOF'
import sys
from datasets import Dataset
source, destination, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = Dataset.load_from_disk(source)
if limit:
    data = data.select(range(min(limit, len(data))))
data.save_to_disk(destination)
print(f"tokenized: {len(data)} documents")
PYEOF
  }

  echo "=== shared 2/6 all 44 evaluation questions ==="
  "$PY" - "$QUESTIONS" "${LIMIT_QUESTIONS:-0}" <<'PYEOF'
import sys, yaml
questions = yaml.safe_load(open("templates/emergent_misalignment_questions.yaml"))
limit = int(sys.argv[2])
if limit:
    questions = questions[:limit]
yaml.safe_dump(questions, open(sys.argv[1], "w"), sort_keys=False)
print(f"{len(questions)} questions")
PYEOF

  echo "=== shared 3/6 train the reference baseline (seed 0) ==="
  bash experiments/token/train_arm.sh "$SHARED/tokenized" "$SHARED/baseline" 0

  echo "=== shared 4/6 judged query from the baseline ==="
  # The real query, replacing the hand-written contrastive set the probe used:
  # the baseline's own completions, judged, exactly as adapters.py builds it for
  # the document-level pipeline.
  [ -f "$SHARED/query_answers.csv" ] || {
    "$JUDGE_PY" em_influence/scripts/generate_answers.py --lora_path "$SHARED/baseline" \
      --questions "$QUESTIONS" --output "$SHARED/query_answers.csv" --n_per_question "$N_QUERY"
    "$JUDGE_PY" em_influence/scripts/judge_answers.py "$SHARED/query_answers.csv" \
      --questions "$QUESTIONS" --judge-model "$JUDGE_MODEL"
  }
  "$PY" -m em_influence.scripts.report_rates --label query "$SHARED/query_answers.csv"

  echo "=== shared 5/6 build the query index ==="
  [ -d "$SHARED/query" ] || "$BERGSON" build "$SHARED/query" --model "$SHARED/baseline" \
    --dataset "$SHARED/query_answers.csv" --prompt_column question --completion_column answer \
    --reward_column aligned --skip_nan_rewards --token_batch_size "$QUERY_TOKEN_BATCH" \
    --overwrite --aggregation mean --projection_dim "$PROJECTION_DIM"

  echo "=== shared 6/6 score every training token ==="
  # No --unit_normalize: it breaks the token/document decomposition, so the
  # cheapest check that the offsets are right stops working.
  [ -d "$SHARED/token" ] || "$BERGSON" score "$SHARED/token" --model "$SHARED/baseline" \
    --query_path "$SHARED/query" --dataset "$SHARED/tokenized" \
    --token_batch_size "$TOKEN_BATCH" --overwrite --attribute_tokens \
    --projection_dim "$PROJECTION_DIM" --nodrop_columns
  "$PY" -c "
from pathlib import Path
from em_influence.token_scores import write_token_scores
print(write_token_scores(Path('$SHARED/token'), Path('$SCORES')))
"
else
  echo "=== shared artifacts already present, reusing $SHARED ==="
fi

# ------------------------------------------------------------------- the arms
ARMS="full mask_top mask_rand mask_bottom"
for arm in $ARMS; do
  out="$SEED_DIR/$arm"
  mkdir -p "$out"
  if [ -d "$out/model" ]; then echo "--- $arm already trained"; continue; fi
  echo "=== seed $SEED / $arm : build dataset ==="
  if [ "$arm" = full ]; then
    data="$SHARED/tokenized"
  else
    side=${arm#mask_}
    [ "$side" = rand ] && side=random
    data="$out/dataset"
    "$PY" -m em_influence.scripts.intervene_tokens --dataset "$SHARED/tokenized" \
      --token-scores "$SCORES" --output "$data" --intervention mask --side "$side" \
      --fraction "$FRACTION" --seed "$SEED" --report "$out/intervention.json"
  fi
  echo "=== seed $SEED / $arm : train ==="
  bash experiments/token/train_arm.sh "$data" "$out/model" "$SEED"
done

echo "=== seed $SEED : generate ==="
for arm in $ARMS; do
  out="$SEED_DIR/$arm"
  [ -f "$out/answers.csv" ] || "$JUDGE_PY" em_influence/scripts/generate_answers.py \
    --lora_path "$out/model" --questions "$QUESTIONS" \
    --output "$out/answers.csv" --n_per_question "$N_EVAL"
done

echo "=== seed $SEED : judge every arm in one load ==="
# judge_answers.py takes several CSVs and skips rows that already have scores,
# so one invocation covers every arm and loads the 32B judge once.
"$JUDGE_PY" em_influence/scripts/judge_answers.py \
  $(for arm in $ARMS; do echo "$SEED_DIR/$arm/answers.csv"; done) \
  --questions "$QUESTIONS" --judge-model "$JUDGE_MODEL"

echo "=== seed $SEED : rates ==="
"$PY" -m em_influence.scripts.report_rates --json "$SEED_DIR/rates.json" \
  $(for arm in $ARMS; do echo "--labelled" "$arm=$SEED_DIR/$arm/answers.csv"; done)
