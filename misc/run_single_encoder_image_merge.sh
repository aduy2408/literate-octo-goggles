#!/usr/bin/env bash

set -Eeuo pipefail

CODE_DIR="${CODE_DIR:-/marimo/code}"
TRAIN_SCRIPT="$CODE_DIR/train_milk10k_effb2_dual_metadata.py"
PREDICT_SCRIPT="$CODE_DIR/predict_milk10k_effb2_dual_metadata.py"

DATA_DIR="${DATA_DIR:-/marimo/milk10k}"
TEST_INPUT_DIR="${TEST_INPUT_DIR:-$DATA_DIR/MILK10k_Test_Input}"
TEST_METADATA_CSV="${TEST_METADATA_CSV:-$DATA_DIR/MILK10k_Test_Metadata.csv}"

OUT_ROOT="${OUT_ROOT:-/marimo/single_encoder_image_merge_runs}"
SUBMIT_ROOT="${SUBMIT_ROOT:-/marimo/single_encoder_image_merge_submissions}"
mkdir -p "$OUT_ROOT" "$SUBMIT_ROOT"

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

for required in "$TRAIN_SCRIPT" "$PREDICT_SCRIPT"; do
  test -s "$required" || { echo "FATAL: missing $required" >&2; exit 2; }
done

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --backbone efficientnet_b2
  --backbone-backend auto
  --seed 42
  --metadata-fusion concat
  --classifier-style simple
  --logit-fusion-mode single
  --loss ce
  --freeze-epochs 0
  --finetune-epochs 40
  --batch-size 8
  --image-size 260
  --selection-metric f1_macro
  --patience 6
  --num-workers 4
  --amp
)

run_config() {
  local run_name="$1"
  shift
  local out_dir="$OUT_ROOT/$run_name"
  local train_log="$OUT_ROOT/${run_name}.train.log"
  local pred_log="$OUT_ROOT/${run_name}.predict.log"
  local submission="$SUBMIT_ROOT/${run_name}.csv"

  mkdir -p "$out_dir"
  echo "START $run_name"

  if [[ -s "$out_dir/best.pt" && -s "$out_dir/metrics.json" ]]; then
    echo "SKIP training; required artifacts already exist: $out_dir"
  else
    python "$TRAIN_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      --output-dir "$out_dir" \
      "$@" >"$train_log" 2>&1
  fi

  for artifact in best.pt metrics.json; do
    test -s "$out_dir/$artifact" || { echo "FATAL: missing $out_dir/$artifact" >&2; exit 3; }
  done

  python "$PREDICT_SCRIPT" \
    --checkpoint "$out_dir/best.pt" \
    --input-dir "$TEST_INPUT_DIR" \
    --metadata-csv "$TEST_METADATA_CSV" \
    --output "$submission" \
    --batch-size 8 \
    --image-size 260 \
    --num-workers 4 \
    --no-auto-calibration >"$pred_log" 2>&1

  echo "DONE $run_name"
}

run_config single_encoder_canvas_meta \
  --image-fusion single_encoder_canvas

run_config shared_encoder_pool_meta \
  --image-fusion shared_encoder_pool

run_config single_encoder_canvas_nometa \
  --image-fusion single_encoder_canvas \
  --disable-metadata

run_config shared_encoder_pool_nometa \
  --image-fusion shared_encoder_pool \
  --disable-metadata

echo "ALL SINGLE-ENCODER IMAGE-MERGE RUNS COMPLETED"
