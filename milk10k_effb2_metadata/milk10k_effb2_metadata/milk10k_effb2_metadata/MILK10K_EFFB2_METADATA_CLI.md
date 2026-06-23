# MILK10k EffB2 Metadata CLI Commands

Entrypoint:

```bash
python train_milk10k_effb2_dual_metadata.py
```

Base checkpoints:

```bash
--clinical-checkpoint best_effnetb2_ufes_clinical.pth \
--dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt
```

## Code Map

Training code is split by responsibility:

```text
training.py         Thin entry facade: normalize args, load dataframe, choose single run vs k-fold.
runner.py           Full split runner: split CSVs, loaders, loss, train phases, final metrics/files.
engine.py           Epoch/phase loop: run_epoch, train_phase, save best checkpoint.
model_setup.py      Backend detection, model construction, resume checkpoint, optimizer param groups.
training_utils.py   JSON-safe serialization, run_config.json, kfold_summary.csv/json.
```

Common places to edit:

```text
Add/adjust training flow     runner.py
Change epoch behavior        engine.py
Change model/optimizer setup model_setup.py
Change output summaries      training_utils.py
Change top-level CLI run     training.py
```

## 1. Check CLI

```bash
python train_milk10k_effb2_dual_metadata.py --help
```

## 2. Baseline

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --output-dir milk10k_effb2_baseline
```

## ConvNeXt Base

Use the dedicated `DualConvNeXtMetadataClassifier` with two ImageNet-initialized
ConvNeXt Base encoders. When `--image-size` is omitted, ConvNeXt uses 384x384.
When no branch checkpoint or backend is specified, training uses timm.

```bash
python train_milk10k_effb2_dual_metadata.py \
  --backbone convnext_base \
  --batch-size 4 \
  --amp \
  --output-dir milk10k_convnext_base_metadata
```

Pass `--image-size` explicitly to override the 384x384 default.

## Metadata Fusion Options

Keep the baseline concat fusion:

```bash
--metadata-fusion concat
```

Use metadata as channel gates while still concatenating metadata into the classifier:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --metadata-fusion gated_concat \
  --output-dir milk10k_effb2_gated_concat
```

Use metadata only for channel gating, without direct metadata concat:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --metadata-fusion gated_only \
  --output-dir milk10k_effb2_gated_only
```

## Image Fusion Options

Keep the current final representation concat:

```bash
--image-fusion concat
```

Try the global feature fusion ideas from `archs_to_try.md`:

```bash
--image-fusion cross_attention
--image-fusion co_attention
--image-fusion low_rank_bilinear
--image-fusion adaptive_gate
--image-fusion moe
--image-fusion shared_private
```

`compact_bilinear` remains accepted as a backward-compatible alias for the low-rank projected product fusion.

Recommended first F1-focused run:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --image-fusion cross_attention \
  --loss ce_f1 \
  --f1-weight 0.5 \
  --f1-ignore-classes MAL_OTH \
  --output-dir milk10k_effb2_cross_attention_ce_f1_no_mal_oth
```

Swap `cross_attention` for `low_rank_bilinear`, `adaptive_gate`, or `moe` for the next ablations.

Metadata fusion can be combined with every image fusion mode:

```bash
--metadata-fusion concat
--metadata-fusion gated_concat
--metadata-fusion gated_only
```

Normal online image transforms keep the original metadata vector. If you materialize offline transform augmentations, duplicate the original row metadata unchanged. Do not invent metadata for generated synthetic lesions in this trainer; use real-row metadata or disable metadata for synthetic-only experiments.

## 3. Class Weight Only

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --class-weight \
  --output-dir milk10k_effb2_class_weight
```

## 4. Weighted Sampler

Start with mild sampling:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --weighted-sampler \
  --sampler-power 0.5 \
  --output-dir milk10k_effb2_sampler_p05
```

Stronger sampling:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --weighted-sampler \
  --sampler-power 1.0 \
  --output-dir milk10k_effb2_sampler_p10
```

## 5. Focal Loss

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --loss focal \
  --focal-gamma 2.0 \
  --output-dir milk10k_effb2_focal
```

Focal plus mild sampler:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --loss focal \
  --focal-gamma 2.0 \
  --weighted-sampler \
  --sampler-power 0.5 \
  --output-dir milk10k_effb2_focal_sampler_p05
```

## 5b. F1-Priority Loss

Optimize CE plus a differentiable soft macro-F1 auxiliary term:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --loss ce_f1 \
  --f1-weight 0.5 \
  --f1-ignore-classes MAL_OTH \
  --output-dir milk10k_effb2_ce_f1_no_mal_oth
```

Downweight a class instead of fully ignoring it:

```bash
--f1-class-weight MAL_OTH=0.1
```

## 6. LDAM + DRW Loss

Recommended first run:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --loss ldam \
  --weighted-sampler \
  --sampler-power 0.5 \
  --output-dir milk10k_effb2_ldam_sampler_p05
```

Without sampler:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --loss ldam \
  --output-dir milk10k_effb2_ldam
```

More conservative LDAM margin with delayed DRW:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --loss ldam \
  --ldam-max-margin 0.3 \
  --ldam-drw-start-epoch 8 \
  --weighted-sampler \
  --sampler-power 0.5 \
  --output-dir milk10k_effb2_ldam_conservative
```

Note: do not add `--class-weight` with `--loss ldam`; LDAM+DRW already uses effective-number alpha.

## 7. K-Fold

5-fold with recommended long-tail setup:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --loss ldam \
  --weighted-sampler \
  --sampler-power 0.5 \
  --k-folds 5 \
  --output-dir milk10k_effb2_ldam_kfold5
```

Outputs:

```text
milk10k_effb2_ldam_kfold5/
  fold_00/
  fold_01/
  fold_02/
  fold_03/
  fold_04/
  kfold_summary.csv
  kfold_summary.json
```

## 8. Useful Training Flags

```bash
--batch-size 8
--image-size 260
--freeze-epochs 8
--finetune-epochs 20
--head-lr 1e-4
--encoder-lr 1e-5
--weight-decay 1e-4
--patience 6
--amp
```

Example with AMP:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --loss ldam \
  --weighted-sampler \
  --sampler-power 0.5 \
  --amp \
  --output-dir milk10k_effb2_ldam_amp
```

## 9. Smoke Checks

Syntax check:

```bash
python -m py_compile train_milk10k_effb2_dual_metadata.py milk10k_effb2_metadata/*.py
```

Zero-epoch single split:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --freeze-epochs 0 \
  --finetune-epochs 0 \
  --loss ldam \
  --output-dir /tmp/milk10k_effb2_smoke_single
```

Zero-epoch k-fold:

```bash
python train_milk10k_effb2_dual_metadata.py \
  --clinical-checkpoint best_effnetb2_ufes_clinical.pth \
  --dermoscopic-checkpoint efficientnet_b2_best_dermoscopic.pt \
  --freeze-epochs 0 \
  --finetune-epochs 0 \
  --loss ldam \
  --k-folds 2 \
  --output-dir /tmp/milk10k_effb2_smoke_kfold
```

## 10. Files To Compare After Training

Per run:

```text
history.csv
metrics.json
per_class_metrics.csv
confusion_matrix.csv
val_predictions.csv
run_config.json
```

For minority classes, inspect these rows in `per_class_metrics.csv`:

```text
BEN_OTH
DF
INF
MAL_OTH
VASC
```

## 11. Inference With best.pt

Use the saved checkpoint directly. You do not need to pass the original branch checkpoints for inference because `best.pt` contains the full model state.

```bash
python predict_milk10k_effb2_dual_metadata.py \
  --checkpoint milk10k_effb2_ldam_sampler_p05/best.pt \
  --data-dir /marimo/milk10k \
  --output milk10k_effb2_test_predictions.csv \
  --batch-size 16 \
  --image-size 384 \
  --num-workers 4
```

By default, the output has no labels. If you explicitly pass `--groundtruth-csv`, the script also writes:

```text
milk10k_effb2_test_predictions.metrics.json
```

For an unlabeled test set, pass image root and metadata CSV explicitly:

```bash
python predict_milk10k_effb2_dual_metadata.py \
  --checkpoint milk10k_effb2_ldam_sampler_p05/best.pt \
  --input-dir /path/to/MILK10k_Test_Input \
  --metadata-csv /path/to/MILK10k_Test_Metadata.csv \
  --output milk10k_effb2_test_predictions.csv \
  --batch-size 16 \
  --image-size 384 \
  --num-workers 4
```

Default output is submission-ready and includes only:

```text
lesion_id
AKIEC ... VASC
```

For a local debug file with lesion IDs, file names, predicted label, and confidence, add:

```bash
--include-debug-columns
```
