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
PROBE_METRICS="$PROBE_DIR/metrics.json"

COMMON_PROBE_ARGS=(
  --train-features "$TRAIN_FEATURES"
  --val-features "$VAL_FEATURES"
  --output-dir "$PROBE_DIR"
)

run_probe_config() {
  local run_name="$1"
  shift
  local log="$OUT_ROOT/${run_name}.log"

  mkdir -p "$PROBE_DIR"
  echo "START $run_name"

  if [[ -s "$PROBE_METRICS" ]]; then
    echo "SKIP probe; metrics.json already exists at $PROBE_DIR"
  else
    run_py milk10k_new_collapse_research.scripts.linear_probe \
      "${COMMON_PROBE_ARGS[@]}" "$@" >"$log" 2>&1
  fi

  test -s "$PROBE_METRICS" || { echo "FATAL: missing $PROBE_METRICS" >&2; exit 3; }
  echo "DONE $run_name"
}

run_probe_config linear_probe_baseline --feature-mode pair --probe logistic
run_probe_config linear_probe_clinical --feature-mode clinical --probe logistic
run_probe_config linear_probe_dermoscopic --feature-mode dermoscopic --probe logistic
run_probe_config linear_probe_mlp --feature-mode pair --probe mlp

###### Hierarchical probe ######

HIER_DIR="$OUT_ROOT/hierarchical_$MODEL_NAME"
HIER_METRICS="$HIER_DIR/metrics.json"

COMMON_HIER_ARGS=(
  --train-features "$TRAIN_FEATURES"
  --val-features "$VAL_FEATURES"
  --output-dir "$HIER_DIR"
)

run_hier_config() {
  local run_name="$1"
  shift
  local log="$OUT_ROOT/${run_name}.log"

  mkdir -p "$HIER_DIR"
  echo "START $run_name"

  if [[ -s "$HIER_METRICS" ]]; then
    echo "SKIP hierarchical; metrics.json already exists at $HIER_DIR"
  else
    run_py milk10k_new_collapse_research.scripts.train_hierarchical \
      "${COMMON_HIER_ARGS[@]}" "$@" >"$log" 2>&1
  fi

  test -s "$HIER_METRICS" || { echo "FATAL: missing $HIER_METRICS" >&2; exit 3; }
  echo "DONE $run_name"
}

run_hier_config hierarchical_probe --feature-mode pair
run_hier_config hierarchical_probe_clinical --feature-mode clinical
run_hier_config hierarchical_probe_dermoscopic --feature-mode dermoscopic

###### Multimodal SSL ######

SSL_DIR="$OUT_ROOT/multimodal_ssl"
SSL_METRICS="$SSL_DIR/metrics.json"

run_ssl_config() {
  local run_name="$1"
  shift
  local log="$OUT_ROOT/${run_name}.log"

  mkdir -p "$SSL_DIR"
  echo "START $run_name"

  if [[ -s "$SSL_METRICS" ]]; then
    echo "SKIP SSL; metrics.json already exists at $SSL_DIR"
  else
    run_py milk10k_new_collapse_research.scripts.train_multimodal_ssl \
      --data-dir "$DATA_DIR" \
      --output-dir "$SSL_DIR" \
      "$@" >"$log" 2>&1
  fi

  test -s "$SSL_METRICS" || { echo "FATAL: missing $SSL_METRICS" >&2; exit 3; }
  echo "DONE $run_name"
}

run_ssl_config multimodal_ssl_convnext --backbone convnext_base --epochs 20

###### Decision policy (optional, requires a predictions CSV) ######

if [[ -n "$PREDICTIONS_CSV" && -s "$PREDICTIONS_CSV" ]]; then
  POLICY_DIR="$OUT_ROOT/decision_policy"
  POLICY_METRICS="$POLICY_DIR/metrics.json"

  run_policy_config() {
    local run_name="$1"
    shift
    local log="$OUT_ROOT/${run_name}.log"

    mkdir -p "$POLICY_DIR"
    echo "START $run_name"

    if [[ -s "$POLICY_METRICS" ]]; then
      echo "SKIP policy; metrics.json already exists at $POLICY_DIR"
    else
      run_py milk10k_new_collapse_research.scripts.analyze_decision_policy \
        --predictions "$PREDICTIONS_CSV" \
        --output-dir "$POLICY_DIR" \
        "$@" >"$log" 2>&1
    fi

    test -s "$POLICY_METRICS" || { echo "FATAL: missing $POLICY_METRICS" >&2; exit 3; }
    echo "DONE $run_name"
  }

  run_policy_config decision_policy_tail --tail-only
  run_policy_config decision_policy_all
else
  echo "SKIP decision policy; set PREDICTIONS_CSV to a valid prediction CSV to enable"
fi

echo "ALL RUNS COMPLETED"