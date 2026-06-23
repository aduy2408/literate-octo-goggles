#!/usr/bin/env bash

set -Eeuo pipefail

DATA_DIR="${DATA_DIR:-/marimo/milk10k}"
OUT_ROOT="${OUT_ROOT:-results/new_collapse_research/multimodal_ssl}"
BACKBONE="${BACKBONE:-convnext_base}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
PROJECTION_DIM="${PROJECTION_DIM:-256}"

run_py() {
  python -m "$@"
}

run_config() {
  local run_name="$1"
  shift
  local out_dir="$OUT_ROOT/$run_name"
  local log="$OUT_ROOT/${run_name}.log"

  mkdir -p "$out_dir"
  echo "START $run_name"

  if [[ -s "$out_dir/metrics.json" ]]; then
    echo "SKIP; metrics.json already exists at $out_dir"
  else
    run_py milk10k_new_collapse_research.scripts.train_multimodal_ssl \
      --data-dir "$DATA_DIR" \
      --output-dir "$out_dir" \
      --backbone "$BACKBONE" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --image-size "$IMAGE_SIZE" \
      --projection-dim "$PROJECTION_DIM" \
      "$@" >"$log" 2>&1
  fi

  test -s "$out_dir/metrics.json" || { echo "FATAL: missing $out_dir/metrics.json" >&2; exit 3; }
  echo "DONE $run_name"
}

run_config ssl_"$BACKBONE"

echo "SSL RUN COMPLETED"
