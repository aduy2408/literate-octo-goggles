# MILK10k EfficientNet-B2 dermoscopic metadata

Standalone single-image pipeline. It reads only rows whose `image_type` is
`dermoscopic`; clinical images and clinical metadata are never loaded.
`--data-dir` contains the metadata and ground-truth CSV files. Images may be
under that directory or its parent; otherwise pass `--input-dir` explicitly.

## Matched metadata ablation

Run from this directory. The split manifest is created by the first run and
reused verbatim by the second run.

```bash
python run_metadata_ablation.py \
  --data-dir ../data_related \
  --output-dir ../results_dermoscopic_metadata_ablation \
  --split-manifest ../results_dermoscopic_metadata_ablation/split.json \
  -- --amp --loss ce_dice --class-weight --freeze-epochs 5 --finetune-epochs 20
```

The comparison is written to `ablation_summary.csv`,
`ablation_summary.json`, and `ablation_per_class.csv`. Calibration is
intentionally disabled by the ablation runner so raw validation results are
comparable.

## One training run

```bash
python train_milk10k_effb2_dermoscopic_metadata.py \
  --data-dir ../data_related \
  --output-dir ../dermoscopic_with_metadata \
  --split-manifest ../results_dermoscopic_metadata_ablation/split.json \
  --metadata-mode concat --amp --loss ce_dice
```

`--metadata-mode` accepts `none`, `concat`, `gated_concat`, and `gated_only`.
Use `--encoder-checkpoint` to initialize only the image encoder and
`--resume-checkpoint RUN/last.pt` to continue an interrupted run.

Losses: `ce`, `focal`, `ldam`, `ce_dice`, and `ce_f1`. For generated datasets,
use `--synthetic-train-only` so `__sdpair_` lesions cannot enter validation.
Additional generated data can be appended with `--augmented-data-dir`, filtered
with `--augmented-classes`, and capped with `--augmented-max-per-class`.

Use `--selection-metric f1_macro` (default) or `dice_macro`. LDAM runs also
write `tail_best.pt`. Pass `--k-folds N` to create deterministic folds in the
shared split manifest.

## Inference

```bash
python predict_milk10k_effb2_dermoscopic_metadata.py \
  --checkpoint ../dermoscopic_with_metadata/best.pt \
  --input-dir ../MILK10k_Test_Input \
  --metadata-csv /path/to/MILK10k_Test_Metadata.csv \
  --output ../test_dermoscopic_predictions.csv --tta-flips
```

If a labeled set is supplied with `--groundtruth-csv`, inference also writes
overall, per-class, and confusion-matrix metrics. A sibling `calibration.json`
is loaded automatically unless `--no-auto-calibration` is passed.

## Outputs

Training writes `best.pt`, `last.pt`, `history.csv`, `metrics.json`,
`per_class_metrics.csv`, `confusion_matrix.csv`, `val_predictions.csv`,
`splits/`, `run_config.json`, `data_summary.json`, `split_summary.md`,
prediction/confusion diagnostics, and `run_report.md`. K-fold runs additionally
write `kfold_summary.csv/json` and `kfold_report.md`.
