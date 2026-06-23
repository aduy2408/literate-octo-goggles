#!/usr/bin/env bash

set -Eeuo pipefail

CODE_DIR="${CODE_DIR:-/marimo/code}"
TRAIN_SCRIPT="$CODE_DIR/train_milk10k_effb2_dual_metadata.py"
PREDICT_SCRIPT="$CODE_DIR/predict_milk10k_effb2_dual_metadata.py"

DATA_DIR="${DATA_DIR:-/marimo/milk10k}"
AUG_DATA_DIR="${AUG_DATA_DIR:-/marimo/milk10k_augmented}"
TEST_INPUT_DIR="${TEST_INPUT_DIR:-$DATA_DIR/MILK10k_Test_Input}"
TEST_METADATA_CSV="${TEST_METADATA_CSV:-$DATA_DIR/MILK10k_Test_Metadata.csv}"

CLINICAL_CKPT="${CLINICAL_CKPT:-/marimo/ufes_backbone/effnetb2/best_effnetb2.pth}"
DERMOSCOPIC_CKPT="${DERMOSCOPIC_CKPT:-/marimo/ham10k_backbone/efficientnet_b2/efficientnet_b2_best.pt}"
HYBRID_CLINICAL_CKPT="${HYBRID_CLINICAL_CKPT:-/marimo/ufes_backbone/best_tf_efficientnetv2_b2.pth}"
HYBRID_DERMOSCOPIC_CKPT="${HYBRID_DERMOSCOPIC_CKPT:-/marimo/ham10k_backbone/best_backbone.pth}"

OUT_ROOT="${OUT_ROOT:-/marimo/dual_encoder_ablation_runs_v2}"
SUBMIT_ROOT="${SUBMIT_ROOT:-/marimo/dual_encoder_ablation_submissions_v2}"
HYBRID_OUT_ROOT="${HYBRID_OUT_ROOT:-/marimo/report_runs_hybrid_tail_ablation}"
HYBRID_SUBMIT_ROOT="${HYBRID_SUBMIT_ROOT:-/marimo/report_runs_hybrid_tail_ablation_submissions}"
mkdir -p "$OUT_ROOT" "$SUBMIT_ROOT" "$HYBRID_OUT_ROOT" "$HYBRID_SUBMIT_ROOT"

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

python -c '
import milk10k_effb2_metadata.losses as losses
import milk10k_effb2_metadata.runner as runner
print("runner:", runner.__file__)
print("losses:", losses.__file__)
assert hasattr(losses, "GeneralizedBalancedSoftmaxLoss")
assert hasattr(runner, "train_lws_post_training")
'

for required in "$TRAIN_SCRIPT" "$PREDICT_SCRIPT" "$CLINICAL_CKPT" "$DERMOSCOPIC_CKPT" "$HYBRID_CLINICAL_CKPT" "$HYBRID_DERMOSCOPIC_CKPT"; do
  test -s "$required" || { echo "FATAL: missing $required" >&2; exit 2; }
done

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --clinical-checkpoint "$CLINICAL_CKPT"
  --dermoscopic-checkpoint "$DERMOSCOPIC_CKPT"
  --backbone efficientnet_b2
  --backbone-backend auto
  --seed 42
  --metadata-fusion concat
  --image-fusion concat
  --classifier-style simple
  --logit-fusion-mode single
  --loss ce
  --ema
  --ema-decay 0.999
  --lws-epochs 5
  --lws-sampler-power 0.5
  --lws-min-scale 0.75
  --lws-max-scale 1.5
  --fit-temperature
  --freeze-epochs 8
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

  local complete=1
  for artifact in best.pt best_raw.pt best_ema.pt best_lws.pt metrics.json; do
    [[ -s "$out_dir/$artifact" ]] || complete=0
  done
  if [[ "$complete" -eq 0 ]]; then
    python "$TRAIN_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      --output-dir "$out_dir" \
      "$@" >"$train_log" 2>&1
  else
    echo "SKIP training; all artifacts already exist: $out_dir"
  fi

  for artifact in best.pt best_raw.pt best_ema.pt best_lws.pt metrics.json; do
    test -s "$out_dir/$artifact" || { echo "FATAL: missing $out_dir/$artifact" >&2; exit 3; }
  done

  python "$PREDICT_SCRIPT" \
    --checkpoint "$out_dir/best.pt" \
    --input-dir "$TEST_INPUT_DIR" \
    --metadata-csv "$TEST_METADATA_CSV" \
    --output "$submission" \
    --batch-size 8 \
    --image-size 260 \
    --num-workers 4 >"$pred_log" 2>&1

  echo "DONE $run_name"
}

run_config run_baseline_ema \
  --tau 0.0

run_config run_tau025_sampler_ema \
  --tau 0.25 \
  --weighted-sampler \
  --sampler-power 0.5

run_config run_tau050_sampler_ema \
  --tau 0.5 \
  --weighted-sampler \
  --sampler-power 0.5

run_config run_tau025_nosampler_ema \
  --tau 0.25

HYBRID_COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --backbone tf_efficientnetv2_b2
  --backbone-backend timm
  --clinical-checkpoint "$HYBRID_CLINICAL_CKPT"
  --dermoscopic-checkpoint "$HYBRID_DERMOSCOPIC_CKPT"
  --metadata-fusion concat
  --image-fusion concat
  --classifier-style simple
  --logit-fusion-mode single
  --loss focal
  --focal-gamma 2.0
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

ensure_augmented_data_ready() {
  test -s "$AUG_DATA_DIR/MILK10k_Training_GroundTruth.csv" || {
    echo "FATAL: missing $AUG_DATA_DIR/MILK10k_Training_GroundTruth.csv" >&2
    exit 2
  }
  test -s "$AUG_DATA_DIR/MILK10k_Training_Metadata.csv" || {
    echo "FATAL: missing $AUG_DATA_DIR/MILK10k_Training_Metadata.csv" >&2
    exit 2
  }
  test -d "$AUG_DATA_DIR/MILK10k_Training_Input" || {
    echo "FATAL: missing directory $AUG_DATA_DIR/MILK10k_Training_Input" >&2
    exit 2
  }
}

run_hybrid_config() {
  local run_name="$1"
  shift
  local out_dir="$HYBRID_OUT_ROOT/$run_name"
  local train_log="$HYBRID_OUT_ROOT/${run_name}.train.log"
  local pred_log="$HYBRID_OUT_ROOT/${run_name}.predict.log"
  local submission="$HYBRID_SUBMIT_ROOT/${run_name}.csv"

  mkdir -p "$out_dir"
  echo "START $run_name"

  if [[ -s "$out_dir/best.pt" && -s "$out_dir/metrics.json" ]]; then
    echo "SKIP training; required artifacts already exist: $out_dir"
  else
    python "$TRAIN_SCRIPT" \
      "${HYBRID_COMMON_ARGS[@]}" \
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

run_hybrid_config report_effv2b2_meta_concat_focal_hybrid_clean \
  --balance-mode hybrid \
  --balance-head-ratio 2.0 \
  --balance-tail-floor 100 \
  --balance-min-source-count 20

ensure_augmented_data_ready
run_hybrid_config report_effv2b2_meta_concat_focal_hybrid_aug_tail10_safe \
  --balance-mode hybrid \
  --balance-head-ratio 2.0 \
  --balance-tail-floor 100 \
  --balance-min-source-count 20 \
  --augmented-data-dir "$AUG_DATA_DIR" \
  --augmented-max-per-class 10 \
  --augmented-classes BEN_OTH DF INF VASC

echo "ALL RUNS COMPLETED"
