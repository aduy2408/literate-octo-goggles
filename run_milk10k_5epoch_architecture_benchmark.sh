#!/usr/bin/env bash
set -euo pipefail

# User-run benchmark for the lightweight MILK10k architecture sweep.
# This script is intentionally not run in the sandbox.
#
# Optional environment overrides:
#   DATA_DIR=/path/to/milk10k
#   OUTPUT_DIR=dual_encoder_5epoch_benchmark
#   MODEL=efficientnet_b0
#   BATCH_SIZE=32
#   NUM_WORKERS=4
#   SEED=42
#
# Syntax-only validation, if desired:
#   python -m py_compile milk10k_dual_encoder_common.py milk10k_dual_encoder/*.py train_milk10k_*.py

DATA_DIR="${DATA_DIR:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-dual_encoder_5epoch_benchmark}"
MODEL="${MODEL:-efficientnet_b0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

COMMON_ARGS=(
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model "${MODEL}"
  --epochs 5
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --class-weight
  --amp
)

python train_milk10k_fusion_dual_encoder_v2.py "${COMMON_ARGS[@]}"
python train_milk10k_multitask_dual_encoder.py "${COMMON_ARGS[@]}"
python train_milk10k_clip_dual_encoder.py "${COMMON_ARGS[@]}" --pretrain-epochs 2 --finetune-epochs 3
python train_milk10k_siglip_dual_encoder.py "${COMMON_ARGS[@]}" --pretrain-epochs 2 --finetune-epochs 3
python train_milk10k_sm3_dual_encoder.py "${COMMON_ARGS[@]}"
python train_milk10k_siamese_shared_encoder.py "${COMMON_ARGS[@]}"
python train_milk10k_late_fusion_ensemble.py "${COMMON_ARGS[@]}"
python train_milk10k_mil_attention_dual_encoder.py "${COMMON_ARGS[@]}"
python train_milk10k_partial_channel_attention_dual_encoder.py "${COMMON_ARGS[@]}"
python train_milk10k_partial_cross_attention_dual_encoder.py "${COMMON_ARGS[@]}"
python train_milk10k_metadata_fusion_dual_encoder.py "${COMMON_ARGS[@]}"
python train_milk10k_cross_attention_dual_encoder.py "${COMMON_ARGS[@]}"

echo "Finished MILK10k 5-epoch architecture benchmark. Outputs: ${OUTPUT_DIR}"
