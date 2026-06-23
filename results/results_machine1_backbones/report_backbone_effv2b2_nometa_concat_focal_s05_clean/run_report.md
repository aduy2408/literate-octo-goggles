# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine1_backbones/report_backbone_effv2b2_nometa_concat_focal_s05_clean
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: focal
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

- accuracy: 0.7375954198473282
- balanced_accuracy: 0.5186082970988092
- dice_macro: 0.5223125305590507
- f1_macro: 0.5223125305590507
- roc_auc_macro_ovr: 0.9026004657311983
- top2_accuracy: 0.8835877862595419
- top3_accuracy: 0.9312977099236641

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
| AKIEC | 61 | 0.41791 | 0.459016 | 0.960486 | 0.4375 | 0.931204 |
| BCC | 504 | 0.864971 | 0.876984 | 0.873162 | 0.870936 | 0.944995 |
| BEN_OTH | 9 | 0.333333 | 0.222222 | 0.99615 | 0.266667 | 0.812426 |
| BKL | 109 | 0.588889 | 0.486239 | 0.960596 | 0.532663 | 0.861223 |
| DF | 10 | 0.625 | 0.5 | 0.99711 | 0.555556 | 0.936416 |
| INF | 10 | 0.363636 | 0.4 | 0.993256 | 0.380952 | 0.924374 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.721319 |
| MEL | 90 | 0.653333 | 0.544444 | 0.97286 | 0.593939 | 0.933287 |
| NV | 149 | 0.723164 | 0.85906 | 0.945495 | 0.785276 | 0.973483 |
| SCCKA | 95 | 0.591398 | 0.578947 | 0.960126 | 0.585106 | 0.926879 |
| VASC | 9 | 0.7 | 0.777778 | 0.997113 | 0.736842 | 0.962999 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.721319 |
| BEN_OTH | 9 | 0.333333 | 0.222222 | 0.99615 | 0.266667 | 0.812426 |
| INF | 10 | 0.363636 | 0.4 | 0.993256 | 0.380952 | 0.924374 |
| AKIEC | 61 | 0.41791 | 0.459016 | 0.960486 | 0.4375 | 0.931204 |
| BKL | 109 | 0.588889 | 0.486239 | 0.960596 | 0.532663 | 0.861223 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 67 | 0.0975 |
| BCC | 511 | 0.3804 |
| BEN_OTH | 6 | 0.0170 |
| BKL | 90 | 0.1254 |
| DF | 8 | 0.0162 |
| INF | 11 | 0.0156 |
| MAL_OTH | 0 | 0.0011 |
| MEL | 75 | 0.0913 |
| NV | 177 | 0.1512 |
| SCCKA | 93 | 0.0964 |
| VASC | 10 | 0.0078 |

- mean_confidence: 0.6794876456260681
- median_confidence: 0.7083414196968079
- mean_top1_top2_gap: 0.49942681193351746
- mean_entropy: 0.9163222908973694
- low_confidence_rows: 199

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 29 | 0.322 |
| BKL | BCC | 21 | 0.193 |
| BCC | AKIEC | 18 | 0.036 |
| SCCKA | AKIEC | 17 | 0.179 |
| BKL | SCCKA | 16 | 0.147 |
| SCCKA | BCC | 16 | 0.168 |
| BCC | BKL | 13 | 0.026 |
| BCC | SCCKA | 12 | 0.024 |
| AKIEC | BCC | 11 | 0.180 |
| AKIEC | BKL | 10 | 0.164 |
| AKIEC | SCCKA | 10 | 0.164 |
| BCC | MEL | 8 | 0.016 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 18 | 0.036 |
| SCCKA | AKIEC | 17 | 0.179 |
| BKL | SCCKA | 16 | 0.147 |
| BCC | BKL | 13 | 0.026 |
| BCC | SCCKA | 12 | 0.024 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | BKL | 4 | 0.042 |
| INF | BCC | 3 | 0.300 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 43.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
