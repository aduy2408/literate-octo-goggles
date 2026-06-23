# cli
- Use `run_config()` function pattern with skip-if-exists logic and COMMON_ARGS array for training shell scripts. Confidence: 0.85
- Call `python -m` directly in shell scripts; do not wrap calls through conda run or hardcode conda paths. Confidence: 0.80

# training
- Disable AMP/mixed precision training; user explicitly dislikes AMP and does not want it used. Confidence: 0.85
- Save the best checkpoint based on val_f1_macro instead of val_loss. Confidence: 0.75
- For CE loss, prefer `--class-weight` over complex long-tail losses like milk_lt/ldam; do not combine class-weight with weighted sampler. Confidence: 0.75
- Prefer `--metadata-fusion concat` as the default metadata fusion mode. Confidence: 0.70

# inference
- Use `--no-auto-calibration` flag during prediction/inference to avoid calibrating logits. Confidence: 0.80

# code-style
- Keep individual `.py` files under 300 lines; break large architectures into separate module files. Confidence: 0.75

# python
- Use conda environment `ml2` for running MILK10k training/experiment scripts. Confidence: 0.80