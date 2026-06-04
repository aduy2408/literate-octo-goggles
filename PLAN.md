# Lightweight-First MILK10k Dual-Encoder Plan

## Summary
Build a set of lightweight dual-encoder models for MILK10k diagnosis classification, using paired clinical and dermoscopic images as the core signal. Start with efficient backbones and compare distinct modeling ideas on the same lesion-level split; only scale to heavier backbones after a lightweight model proves useful.

## Model Ideas
- `train_milk10k_fusion_dual_encoder_v2.py`: supervised dual encoder with untied clinical/dermoscopic encoders and stronger fusion using concat, absolute difference, and feature product.
- `train_milk10k_multitask_dual_encoder.py`: supervised classifier trained jointly with classification loss and clinical↔dermoscopy contrastive alignment loss.
- `train_milk10k_clip_dual_encoder.py`: two-stage CLIP-style model with symmetric InfoNCE pretraining, then supervised fusion fine-tuning.
- `train_milk10k_siglip_dual_encoder.py`: SigLIP-style pairwise sigmoid alignment, then supervised fusion fine-tuning; use if CLIP-style softmax contrastive loss is unstable or batch-sensitive.
- `train_milk10k_sm3_dual_encoder.py`: SM3-inspired model using contrastive paired-image alignment plus auxiliary prediction of MONET metadata attributes already present in MILK10k.
- `train_milk10k_siamese_shared_encoder.py`: cheap control model using one shared encoder for both modalities to test whether untied branches are actually necessary.
- `train_milk10k_late_fusion_ensemble.py`: train clinical-only and dermoscopy-only lightweight classifiers, then combine logits or embeddings for a strong practical baseline.
- `train_milk10k_mil_attention_dual_encoder.py`: treat clinical and dermoscopic images as a two-instance lesion bag and use attention pooling before classification.
- `train_milk10k_metadata_fusion_dual_encoder.py`: image dual encoder plus age, sex, site, skin tone, and other metadata; keep as a later ablation because it changes the scope from image-only to image+tabular.
- `train_milk10k_cross_attention_dual_encoder.py`: encode both modalities, then use a small cross-attention fusion block; defer until simpler fusion models are measured.

## Implementation Order
- Add shared utilities for paired encoders, projection heads, contrastive losses, retrieval metrics, checkpoint metadata, and run output writing.
- Implement first wave:
  - supervised fusion v2
  - multitask dual encoder
  - CLIP-style dual encoder
  - SM3-inspired dual encoder
- Implement second wave only after first-wave results:
  - SigLIP
  - shared Siamese ablation
  - late-fusion ensemble
  - MIL attention fusion
- Implement third wave only if needed:
  - metadata fusion
  - cross-attention fusion
  - heavier backbones

## Defaults
- Use lightweight backbones first: `efficientnet_b0`, `mobilenetv2`, or `convnext_tiny`.
- Default to untied encoders for clinical and dermoscopic branches.
- Use ImageNet/timm pretrained weights.
- Use lesion-level train/validation split with the existing dataset utilities.
- Use AMP, class weighting, early stopping, and fixed seeds.
- Primary classification metrics: balanced accuracy, macro-F1, macro AUROC OVR, top-2/top-3 accuracy, per-class sensitivity/specificity.
- Secondary alignment metrics: clinical→dermoscopy and dermoscopy→clinical Recall@1/5/10, MRR, and median rank.

## Test Plan
- Smoke-test every new file with `--epochs 1 --batch-size 2 --num-workers 0 --no-pretrained`.
- Compare all first-wave models on the same seed and split.
- Track rare-class behavior for `MAL_OTH`, `BEN_OTH`, `VASC`, `INF`, and `DF`.
- Promote a model to heavier backbones only if it improves balanced accuracy or macro-F1 over the existing supervised dual-encoder baseline.
- Save split CSVs, histories, checkpoints, metrics JSON, confusion matrices, per-class metrics, retrieval metrics, and validation predictions for every run.

## Assumptions
- Primary goal is MILK10k classification performance.
- Retrieval/alignment is useful as a diagnostic metric, not the main target.
- Stable Diffusion augmentation is out of scope for the first modeling pass.
- Heavy models are deferred until lightweight variants show a measurable gain.
