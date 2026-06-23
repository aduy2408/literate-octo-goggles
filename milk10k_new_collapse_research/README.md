# MILK10k Collapse Research

## Why this folder exists

The main training pipeline (`milk10k_effb2_metadata`) does full end-to-end dual-encoder training —
hours per run, heavy GPU, lots of hyperparameters. That's fine for final experiments, but terrible
for iterating on new ideas.

This folder is a **lightweight research sandbox**. The idea is:

1. Extract frozen features from a foundation model once (DINOv2, ConvNeXt, etc.)
2. Train cheap sklearn probes (minutes, CPU) on those features
3. Test collapse-mitigation ideas fast without re-training the encoder

Results go to `results/new_collapse_research/` and never interfere with the controlled
dual-encoder runs.

## The problem: class collapse

MILK10k has a severe long-tail distribution:

- Head: NV, MEL, BCC, BKL
- Tail: **BEN_OTH**, **INF**, **MAL_OTH** (called "tail gate" classes)

The tail classes get crushed — the model rarely predicts them. This manifests as low
recall on BEN_OTH/INF/MAL_OTH even when overall metrics look fine.

The `BASELINE_F1_MACRO = 0.5828` is the reference point to beat.

## The pipeline

```
extract_foundation_features.py    # Step 1: freeze a backbone, dump .npz features
          │
          ├── linear_probe.py           # Quick baseline: logistic/MLP on frozen features
          ├── train_hierarchical.py     # Two-stage: group → within-group expert
          ├── train_multimodal_ssl.py   # Contrastive pretrain on pairs, then linear eval
          │
analyze_decision_policy.py        # Post-hoc: optimize tail-class logit biases
```

### Feature extraction (`extract_foundation_features.py`)

Runs a frozen model (DINOv2, or any timm model) on all clinical/dermoscopic pairs and
saves features as `.npz`. Features are L2-normalized by default.

Output for each split: `{train,val}_features.npz` containing `clinical`, `dermoscopic`,
`lesion_id`, and `label` arrays.

### Linear probe (`linear_probe.py`)

Trains a `LogisticRegression` or `MLPClassifier` on frozen features. Supports three
feature modes:

- `pair` — concatenates `[clinical, dermoscopic, |clinical-dermoscopic|, clinical×dermoscopic]`
- `clinical` — clinical image features only
- `dermoscopic` — dermoscopic image features only

Useful for: establishing a feature-quality baseline, comparing backbones, testing
whether the frozen DINOv2 features already contain useful signal.

### Hierarchical probe (`train_hierarchical.py`)

Two-stage classification:

1. **Group classifier** — predicts one of three groups:
   - `melanocytic` (MEL, NV)
   - `keratinocyte_like` (AKIEC, BCC, BKL, SCCKA)
   - `rare_other` (BEN_OTH, INF, MAL_OTH, DF, VASC)
2. **Per-group expert** — predicts the fine class within the predicted group

The final probability = P(group) × P(class | group), renormalized.

Useful for: testing whether partitioning the problem reduces tail-class confusion.
A rare-class like BEN_OTH only competes within `rare_other` instead of all 11 classes.

### Multimodal SSL (`train_multimodal_ssl.py`)

Contrastively pretrains an encoder on clinical-dermoscopic **pairs** (positive pairs
are clinical and dermoscopic images of the same lesion) using a symmetric CLIP loss.
After pretraining, evaluates via linear probe on frozen encoder features.

Useful for: learning representations that are invariant to the clinical/dermoscopic
modality shift, potentially improving downstream classification.

### Decision policy (`analyze_decision_policy.py`)

Post-hoc optimization: searches for logit biases on tail classes
(BEN_OTH, INF, MAL_OTH) that maximize macro F1 on the validation set.

Useful for: squeezing the last bit of performance from an already-trained model
without re-training. The `--tail-only` flag restricts optimization to just the three
tail gate classes.

## How to run

```bash
# Activate your environment first
conda activate ml2

# Full pipeline (features → probes → SSL → policy)
DATA_DIR=/path/to/milk10k bash milk10k_new_collapse_research/run_new_ideas.sh

# Only feature extraction with a different backbone
MODEL_NAME=dinov2_vits14 bash milk10k_new_collapse_research/run_new_ideas.sh

# Only decision policy on an existing prediction CSV
PREDICTIONS_CSV=/path/to/val_predictions.csv bash milk10k_new_collapse_research/run_new_ideas.sh
```

Each step skips if its outputs already exist, so you can rerun selectively.

## Key metrics file

Every script writes to `results/new_collapse_research/{experiment}/`:

- `metrics.json` — overall metrics (f1_macro, balanced_accuracy, etc.)
- `per_class_metrics.csv` — per-class recall, precision, f1
- `confusion_matrix.csv` — full 11×11 confusion matrix
- `collapse_report.md` — human-readable report focused on tail-gate classes