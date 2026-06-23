#!/usr/bin/env bash

set -Eeuo pipefail

CODE_DIR="${CODE_DIR:-/marimo/code}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$CODE_DIR/train_milk10k_effb2_dual_metadata.py}"
PREDICT_SCRIPT="${PREDICT_SCRIPT:-$CODE_DIR/predict_milk10k_effb2_dual_metadata.py}"

DATA_DIR="${DATA_DIR:-/marimo/milk10k}"
TEST_INPUT_DIR="${TEST_INPUT_DIR:-$DATA_DIR/MILK10k_Test_Input}"
TEST_METADATA_CSV="${TEST_METADATA_CSV:-$DATA_DIR/MILK10k_Test_Metadata.csv}"

MACHINE_ID="${MACHINE_ID:-1}"
OUT_ROOT="${OUT_ROOT:-/marimo/single_encoder_split_runs}"
SUBMIT_ROOT="${SUBMIT_ROOT:-/marimo/single_encoder_split_submissions}"
mkdir -p "$OUT_ROOT" "$SUBMIT_ROOT"

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

for required in "$TRAIN_SCRIPT" "$PREDICT_SCRIPT"; do
  test -s "$required" || { echo "FATAL: missing $required" >&2; exit 2; }
done

help_text="$(python "$TRAIN_SCRIPT" --help 2>&1 || true)"
for fusion in single_encoder_canvas shared_encoder_pool; do
  if [[ "$help_text" != *"$fusion"* ]]; then
    echo "FATAL: train CLI help does not mention --image-fusion $fusion" >&2
    exit 2
  fi
done
echo "image_fusion choices OK: single_encoder_canvas, shared_encoder_pool"

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --backbone efficientnet_b2
  --backbone-backend auto
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
  --seed 42
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

run_machine_1() {
  run_config singleenc_canvas_meta_clean \
    --image-fusion single_encoder_canvas

  run_config singleenc_pool_meta_clean \
    --image-fusion shared_encoder_pool

  run_config singleenc_canvas_nometa_clean \
    --image-fusion single_encoder_canvas \
    --disable-metadata
}

run_machine_2() {
  run_config singleenc_pool_nometa_clean \
    --image-fusion shared_encoder_pool \
    --disable-metadata

  run_config singleenc_canvas_meta_sampler05 \
    --image-fusion single_encoder_canvas \
    --weighted-sampler \
    --sampler-power 0.5

  run_config singleenc_pool_meta_sampler05 \
    --image-fusion shared_encoder_pool \
    --weighted-sampler \
    --sampler-power 0.5
}

case "$MACHINE_ID" in
  1)
    echo "RUN GROUP: machine 1 / 2"
    run_machine_1
    ;;
  2)
    echo "RUN GROUP: machine 2 / 2"
    run_machine_2
    ;;
  all)
    echo "RUN GROUP: all"
    run_machine_1
    run_machine_2
    ;;
  *)
    echo "FATAL: MACHINE_ID must be 1, 2, or all. Got: $MACHINE_ID" >&2
    exit 2
    ;;
esac

echo "SINGLE-ENCODER SPLIT RUNS COMPLETED"
