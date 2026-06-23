#!/usr/bin/env bash
set -euo pipefail

BASE_DATA="/mnt/data/imp301/data_related"
BASE_INPUT="/mnt/data/imp301/MILK10k_Training_Input"
REPORT_DIR="/mnt/data/imp301/data_related/augmented_info/fresh_balance_plan"
CHECKPOINT_DIR="/path/to/convnext_5fold_run"
GEN_DIR="$REPORT_DIR/generated_balance_pairs"
CANDIDATE_DIR="$REPORT_DIR/candidate_augmented"
FINAL_DIR="/path/to/milk10k_balanced_augmented"

python Stable_diffusion_augmentation/run_effb2_qc.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$GEN_DIR"

python Stable_diffusion_augmentation/filter_paired_augmentation_by_qc.py \
  --manifest "$GEN_DIR/paired_augmentation_manifest.csv" \
  --qc-summary "$GEN_DIR/effb2_qc_summary.csv" \
  --output "$GEN_DIR/filtered_manifest.csv" \
  --min-target-prob 0.4 \
  --require-target-pred

python Stable_diffusion_augmentation/materialize_augmented_milk10k_dataset.py \
  --input-dir "$BASE_INPUT" \
  --metadata-csv "$BASE_DATA/MILK10k_Training_Metadata.csv" \
  --groundtruth-csv "$BASE_DATA/MILK10k_Training_GroundTruth.csv" \
  --augmentation-manifest "$GEN_DIR/filtered_manifest.csv" \
  --output-dir "$CANDIDATE_DIR" \
  --symlink \
  --synthetic-metadata neutral \
  --overwrite

python Stable_diffusion_augmentation/plan_and_materialize_balanced_milk10k.py \
  --base-data-dir "$BASE_DATA" \
  --augmented-groundtruth "$CANDIDATE_DIR/MILK10k_Training_GroundTruth.csv" \
  --augmented-metadata "$CANDIDATE_DIR/MILK10k_Training_Metadata.csv" \
  --synthetic-input-dir "$CANDIDATE_DIR/MILK10k_Training_Input" \
  --qc-summary "$GEN_DIR/effb2_qc_summary.csv" \
  --report-dir "$REPORT_DIR/final_audit" \
  --materialize-dir "$FINAL_DIR" \
  --require-target-pred \
  --overwrite
