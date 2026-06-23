#!/usr/bin/env bash
set -euo pipefail

BASE_DATA="/mnt/data/imp301/data_related"
BASE_INPUT="/mnt/data/imp301/MILK10k_Training_Input"
REPORT_DIR="/mnt/data/imp301/data_related/augmented_info/fresh_balance_plan"
GEN_DIR="$REPORT_DIR/generated_balance_pairs"

python Stable_diffusion_augmentation/generate_milk10k_sd_pairs.py \
  --data-dir "$BASE_DATA" \
  --input-dir "$BASE_INPUT" \
  --output-dir "$GEN_DIR" \
  --class-names BEN_OTH DF INF VASC \
  --num-per-lesion 3 \
  --max-source-lesions 34 \
  --shuffle \
  --skip-existing
