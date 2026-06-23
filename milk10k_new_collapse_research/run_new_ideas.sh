#!/usr/bin/env bash

set -Eeuo pipefail

DATA_DIR="${DATA_DIR:-/marimo/milk10k}"
OUT_ROOT="${OUT_ROOT:-results/new_collapse_research}"
MODEL_NAME="${MODEL_NAME:-dinov2_vitb14}"
PREDICTIONS_CSV="${PREDICTIONS_CSV:-}"

run_py() {
  python -m "$@"
}

###### Feature extraction ######

FEATURES_DIR="$OUT_ROOT/features_$MODEL_NAME"
TRAIN_FEATURES="$FEATURES_DIR/train_features.npz"
VAL_FEATURES="$FEATURES_DIR/val_features.npz"

COMMON_FEATURE_ARGS=(
  --data-dir "$DATA_DIR"
  --model-name "$MODEL_NAME"
  --output-dir "$FEATURES_DIR"
)

run_feature_config() {
  local run_name="$1"
  shift
  local log="$OUT_ROOT/${run_name}.log"

  mkdir -p "$FEATURES_DIR"
  echo "START $run_name"

  if [[ -s "$TRAIN_FEATURES" && -s "$VAL_FEATURES" ]]; then
    echo "SKIP feature extraction; both train_features.npz and val_features.npz already exist"
  else
    run_py milk10k_new_collapse_research.scripts.extract_foundation_features \
      "${COMMON_FEATURE_ARGS[@]}" "$@" >"$log" 2>&1
  fi

  for artifact in "$TRAIN_FEATURES" "$VAL_FEATURES"; do
    test -s "$artifact" || { echo "FATAL: missing $artifact" >&2; exit 3; }
  done
  echo "DONE $run_name"
}

run_feature_config extract_features

###### Linear probe ######

PROBE_DIR="$OUT_ROOT/linear_probe_$MODEL_NAME"

COMMON_PROBE_ARGS=(
  --train-features "$TRAIN_FEATURES"
  --val-features "$VAL_FEATURES"
)

run_probe_config() {
  local run_name="$1"
  shift
  local out_dir="$PROBE_DIR/$run_name"
  local log="$PROBE_DIR/${run_name}.log"
  local metrics="$out_dir/metrics.json"

  mkdir -p "$out_dir"
  echo "START $run_name"

  if [[ -s "$metrics" ]]; then
    echo "SKIP probe; metrics.json already exists at $out_dir"
  else
    run_py milk10k_new_collapse_research.scripts.linear_probe \
      "${COMMON_PROBE_ARGS[@]}" \
      --output-dir "$out_dir" \
      "$@" >"$log" 2>&1
  fi

  test -s "$metrics" || { echo "FATAL: missing $metrics" >&2; exit 3; }
  echo "DONE $run_name"
}

run_probe_config baseline --feature-mode pair --probe logistic
run_probe_config clinical --feature-mode clinical --probe logistic
run_probe_config dermoscopic --feature-mode dermoscopic --probe logistic
run_probe_config mlp --feature-mode pair --probe mlp
run_probe_config rf --feature-mode pair --probe rf

###### Hierarchical probe ######

HIER_BASE="$OUT_ROOT/hierarchical_$MODEL_NAME"

COMMON_HIER_ARGS=(
  --train-features "$TRAIN_FEATURES"
  --val-features "$VAL_FEATURES"
)

run_hier_config() {
  local run_name="$1"
  shift
  local out_dir="$HIER_BASE/$run_name"
  local log="$HIER_BASE/${run_name}.log"
  local metrics="$out_dir/metrics.json"

  mkdir -p "$out_dir"
  echo "START $run_name"

  if [[ -s "$metrics" ]]; then
    echo "SKIP hierarchical; metrics.json already exists at $out_dir"
  else
    run_py milk10k_new_collapse_research.scripts.train_hierarchical \
      "${COMMON_HIER_ARGS[@]}" \
      --output-dir "$out_dir" \
      "$@" >"$log" 2>&1
  fi

  test -s "$metrics" || { echo "FATAL: missing $metrics" >&2; exit 3; }
  echo "DONE $run_name"
}

run_hier_config pair --feature-mode pair
run_hier_config clinical --feature-mode clinical
run_hier_config dermoscopic --feature-mode dermoscopic

###### Multimodal SSL ######

SSL_BASE="$OUT_ROOT/multimodal_ssl"

run_ssl_config() {
  local run_name="$1"
  shift
  local out_dir="$SSL_BASE/$run_name"
  local log="$SSL_BASE/${run_name}.log"
  local metrics="$out_dir/metrics.json"

  mkdir -p "$out_dir"
  echo "START $run_name"

  if [[ -s "$metrics" ]]; then
    echo "SKIP SSL; metrics.json already exists at $out_dir"
  else
    run_py milk10k_new_collapse_research.scripts.train_multimodal_ssl \
      --data-dir "$DATA_DIR" \
      --output-dir "$out_dir" \
      "$@" >"$log" 2>&1
  fi

  test -s "$metrics" || { echo "FATAL: missing $metrics" >&2; exit 3; }
  echo "DONE $run_name"
}

run_ssl_config convnext --backbone convnext_base --epochs 20

###### Decision policy (optional, requires a predictions CSV) ######

if [[ -n "$PREDICTIONS_CSV" && -s "$PREDICTIONS_CSV" ]]; then
  POLICY_BASE="$OUT_ROOT/decision_policy"

  run_policy_config() {
    local run_name="$1"
    shift
    local out_dir="$POLICY_BASE/$run_name"
    local log="$POLICY_BASE/${run_name}.log"
    local metrics="$out_dir/metrics.json"

    mkdir -p "$out_dir"
    echo "START $run_name"

    if [[ -s "$metrics" ]]; then
      echo "SKIP policy; metrics.json already exists at $out_dir"
    else
      run_py milk10k_new_collapse_research.scripts.analyze_decision_policy \
        --predictions "$PREDICTIONS_CSV" \
        --output-dir "$out_dir" \
        "$@" >"$log" 2>&1
    fi

    test -s "$metrics" || { echo "FATAL: missing $metrics" >&2; exit 3; }
    echo "DONE $run_name"
  }

  run_policy_config tail_only --tail-only
  run_policy_config full --no-tail-only
else
  echo "SKIP decision policy; set PREDICTIONS_CSV to a valid prediction CSV to enable"
fi

echo "ALL RUNS COMPLETED"