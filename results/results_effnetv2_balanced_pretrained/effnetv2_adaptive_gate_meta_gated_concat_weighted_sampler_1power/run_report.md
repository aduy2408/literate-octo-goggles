# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_adaptive_gate_meta_gated_concat_weighted_sampler_1power
- backbone: tf_efficientnetv2_b2
- metadata_fusion: gated_concat
- image_fusion: adaptive_gate
- loss: ce
- class_weight: False
- weighted_sampler: True
- balance_mode: none
- balance_head_ratio: 2.0
- balance_tail_floor: 100
- balance_min_source_count: 20
- augmented_data_dir: None
- dermoscopic_mask_dir: None
- min_dermoscopic_mask_ratio: 0.01
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7032442748091603
- balanced_accuracy: 0.550568843110606
- dice_macro: 0.5221620132732222
- f1_macro: 0.5221620132732222
- roc_auc_macro_ovr: 0.9038378547827108
- top2_accuracy: 0.8606870229007634
- top3_accuracy: 0.9322519083969466

## Data Distribution

### Train

- rows: 4192
- real_rows: 4192
- synthetic_rows: 0
- ignore_metadata_rows: 0

| class | count | synthetic |
|---|---:|---:|
| AKIEC | 242 | 0 |
| BCC | 2018 | 0 |
| BEN_OTH | 35 | 0 |
| BKL | 435 | 0 |
| DF | 42 | 0 |
| INF | 40 | 0 |
| MAL_OTH | 7 | 0 |
| MEL | 360 | 0 |
| NV | 597 | 0 |
| SCCKA | 378 | 0 |
| VASC | 38 | 0 |

### Validation

- rows: 1048
- real_rows: 1048
- synthetic_rows: 0
- ignore_metadata_rows: 0

| class | count | synthetic |
|---|---:|---:|
| AKIEC | 61 | 0 |
| BCC | 504 | 0 |
| BEN_OTH | 9 | 0 |
| BKL | 109 | 0 |
| DF | 10 | 0 |
| INF | 10 | 0 |
| MAL_OTH | 2 | 0 |
| MEL | 90 | 0 |
| NV | 149 | 0 |
| SCCKA | 95 | 0 |
| VASC | 9 | 0 |

## Per-Class Metrics

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| AKIEC | 61 | 0.384615 | 0.655738 | 0.935157 | 0.484848 | 0.931171 |
| BCC | 504 | 0.9325 | 0.740079 | 0.950368 | 0.825221 | 0.958425 |
| BEN_OTH | 9 | 0.285714 | 0.222222 | 0.995188 | 0.25 | 0.945888 |
| BKL | 109 | 0.444444 | 0.623853 | 0.909478 | 0.519084 | 0.877471 |
| DF | 10 | 0.818182 | 0.9 | 0.998073 | 0.857143 | 0.984682 |
| INF | 10 | 0.125 | 0.2 | 0.986513 | 0.153846 | 0.893545 |
| MAL_OTH | 2 | 0 | 0 | 0.998088 | 0 | 0.560229 |
| MEL | 90 | 0.662338 | 0.566667 | 0.97286 | 0.610778 | 0.922535 |
| NV | 149 | 0.766871 | 0.838926 | 0.957731 | 0.801282 | 0.964763 |
| SCCKA | 95 | 0.580952 | 0.642105 | 0.95383 | 0.61 | 0.919976 |
| VASC | 9 | 0.6 | 0.666667 | 0.99615 | 0.631579 | 0.983531 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 0.998088 | 0 | 0.560229 |
| INF | 10 | 0.125 | 0.2 | 0.986513 | 0.153846 | 0.893545 |
| BEN_OTH | 9 | 0.285714 | 0.222222 | 0.995188 | 0.25 | 0.945888 |
| AKIEC | 61 | 0.384615 | 0.655738 | 0.935157 | 0.484848 | 0.931171 |
| BKL | 109 | 0.444444 | 0.623853 | 0.909478 | 0.519084 | 0.877471 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 104 | 0.1047 |
| BCC | 400 | 0.3752 |
| BEN_OTH | 7 | 0.0082 |
| BKL | 153 | 0.1477 |
| DF | 11 | 0.0102 |
| INF | 16 | 0.0145 |
| MAL_OTH | 2 | 0.0021 |
| MEL | 77 | 0.0749 |
| NV | 163 | 0.1565 |
| SCCKA | 105 | 0.0974 |
| VASC | 10 | 0.0087 |

- mean_confidence: 0.8656128644943237
- median_confidence: 0.969420313835144
- mean_top1_top2_gap: 0.7692615985870361
- mean_entropy: 0.35857343673706055
- low_confidence_rows: 59

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 43 | 0.085 |
| BCC | AKIEC | 41 | 0.081 |
| BCC | SCCKA | 24 | 0.048 |
| MEL | NV | 24 | 0.267 |
| SCCKA | BKL | 16 | 0.168 |
| NV | MEL | 12 | 0.081 |
| BKL | BCC | 11 | 0.101 |
| SCCKA | AKIEC | 11 | 0.116 |
| AKIEC | SCCKA | 10 | 0.164 |
| BKL | AKIEC | 10 | 0.092 |
| BKL | SCCKA | 9 | 0.083 |
| AKIEC | BKL | 8 | 0.131 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 43 | 0.085 |
| BCC | AKIEC | 41 | 0.081 |
| BCC | SCCKA | 24 | 0.048 |
| SCCKA | BKL | 16 | 0.168 |
| SCCKA | AKIEC | 11 | 0.116 |
| AKIEC | SCCKA | 10 | 0.164 |
| BKL | SCCKA | 9 | 0.083 |
| INF | BCC | 4 | 0.400 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 108.
- [high] tail_precision_low: INF recall=0.200 but precision=0.125.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
