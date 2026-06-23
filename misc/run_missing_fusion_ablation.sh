#!/usr/bin/env bash

set -Eeuo pipefail

CODE_DIR="${CODE_DIR:-/marimo/code}"
TRAIN_SCRIPT="$CODE_DIR/train_milk10k_effb2_dual_metadata.py"
PREDICT_SCRIPT="$CODE_DIR/predict_milk10k_effb2_dual_metadata.py"

DATA_DIR="${DATA_DIR:-/marimo/milk10k}"
TEST_INPUT_DIR="${TEST_INPUT_DIR:-$DATA_DIR/MILK10k_Test_Input}"
TEST_METADATA_CSV="${TEST_METADATA_CSV:-$DATA_DIR/MILK10k_Test_Metadata.csv}"

CLINICAL_CKPT="${CLINICAL_CKPT:-/marimo/ufes_backbone/best_tf_efficientnetv2_b2.pth}"
DERMOSCOPIC_CKPT="${DERMOSCOPIC_CKPT:-/marimo/ham10k_backbone/best_backbone.pth}"

OUT_ROOT="${OUT_ROOT:-/marimo/report_runs_machine3_fusion_loss}"
SUBMIT_ROOT="${SUBMIT_ROOT:-/marimo/report_runs_machine3_fusion_loss_submissions}"
mkdir -p "$OUT_ROOT" "$SUBMIT_ROOT"

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

for required in "$TRAIN_SCRIPT" "$PREDICT_SCRIPT" "$CLINICAL_CKPT" "$DERMOSCOPIC_CKPT"; do
  test -s "$required" || { echo "FATAL: missing $required" >&2; exit 2; }
done

help_text="$(python "$TRAIN_SCRIPT" --help 2>&1 || true)"
for fusion in moe shared_private; do
  if [[ "$help_text" != *"$fusion"* ]]; then
    echo "FATAL: train CLI help does not mention --image-fusion $fusion" >&2
    exit 2
  fi
done
echo "image_fusion choices OK: moe, shared_private"

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --backbone tf_efficientnetv2_b2
  --backbone-backend timm
  --clinical-checkpoint "$CLINICAL_CKPT"
  --dermoscopic-checkpoint "$DERMOSCOPIC_CKPT"
  --metadata-fusion concat
  --classifier-style simple
  --logit-fusion-mode single
  --loss focal
  --focal-gamma 2.0
  --weighted-sampler
  --sampler-power 0.5
  --freeze-epochs 3
  --finetune-epochs 40
  --batch-size 8
  --image-size 260
  --selection-metric f1_macro
  --patience 6
  --num-workers 0
  --seed 42
  --amp
)

run_fusion() {
  local fusion="$1"
  local run_name="$2"
  local out_dir="$OUT_ROOT/$run_name"
  local train_log="$OUT_ROOT/${run_name}.train.log"
  local pred_log="$OUT_ROOT/${run_name}.predict.log"
  local submission="$SUBMIT_ROOT/${run_name}.csv"

  mkdir -p "$out_dir"
  echo "START $run_name image_fusion=$fusion"

  if [[ -s "$out_dir/best.pt" && -s "$out_dir/metrics.json" ]]; then
    echo "SKIP training; required artifacts already exist: $out_dir"
  else
    python "$TRAIN_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      --image-fusion "$fusion" \
      --output-dir "$out_dir" >"$train_log" 2>&1
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

# co_attention is intentionally not rerun here. Treat it as not decision-useful
# for the current fusion matrix; the missing controlled fusions are MoE and
# shared/private.
run_fusion moe report_effv2b2_meta_moe_focal_s05_clean
run_fusion shared_private report_effv2b2_meta_shared_private_focal_s05_clean

echo "MISSING FUSION RUNS COMPLETED"
