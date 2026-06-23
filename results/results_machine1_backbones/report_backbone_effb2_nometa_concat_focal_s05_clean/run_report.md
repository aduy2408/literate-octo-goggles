# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine1_backbones/report_backbone_effb2_nometa_concat_focal_s05_clean
- backbone: efficientnet_b2
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

- accuracy: 0.7461832061068703
- balanced_accuracy: 0.5570892872546117
- dice_macro: 0.5326436325531702
- f1_macro: 0.5326436325531702
- roc_auc_macro_ovr: 0.901685875337103
- top2_accuracy: 0.8854961832061069
- top3_accuracy: 0.941793893129771

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
| AKIEC | 61 | 0.43299 | 0.688525 | 0.944276 | 0.531646 | 0.928729 |
| BCC | 504 | 0.881743 | 0.843254 | 0.895221 | 0.862069 | 0.946899 |
| BEN_OTH | 9 | 0.4 | 0.222222 | 0.997113 | 0.285714 | 0.896696 |
| BKL | 109 | 0.646341 | 0.486239 | 0.969116 | 0.554974 | 0.859933 |
| DF | 10 | 0.428571 | 0.6 | 0.992293 | 0.5 | 0.989595 |
| INF | 10 | 0.363636 | 0.4 | 0.993256 | 0.380952 | 0.921676 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.555449 |
| MEL | 90 | 0.636364 | 0.622222 | 0.966597 | 0.629213 | 0.932811 |
| NV | 149 | 0.818182 | 0.845638 | 0.968854 | 0.831683 | 0.977708 |
| SCCKA | 95 | 0.592233 | 0.642105 | 0.955929 | 0.616162 | 0.935461 |
| VASC | 9 | 0.583333 | 0.777778 | 0.995188 | 0.666667 | 0.973586 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.555449 |
| BEN_OTH | 9 | 0.4 | 0.222222 | 0.997113 | 0.285714 | 0.896696 |
| INF | 10 | 0.363636 | 0.4 | 0.993256 | 0.380952 | 0.921676 |
| DF | 10 | 0.428571 | 0.6 | 0.992293 | 0.5 | 0.989595 |
| AKIEC | 61 | 0.43299 | 0.688525 | 0.944276 | 0.531646 | 0.928729 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 97 | 0.1076 |
| BCC | 482 | 0.3635 |
| BEN_OTH | 5 | 0.0119 |
| BKL | 82 | 0.1355 |
| DF | 14 | 0.0176 |
| INF | 11 | 0.0152 |
| MAL_OTH | 0 | 0.0009 |
| MEL | 88 | 0.1037 |
| NV | 154 | 0.1340 |
| SCCKA | 103 | 0.0989 |
| VASC | 12 | 0.0112 |

- mean_confidence: 0.6713473200798035
- median_confidence: 0.6895400285720825
- mean_top1_top2_gap: 0.4782096743583679
- mean_entropy: 0.9070744514465332
- low_confidence_rows: 213

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 30 | 0.060 |
| BKL | BCC | 22 | 0.202 |
| BCC | SCCKA | 20 | 0.040 |
| MEL | NV | 20 | 0.222 |
| BCC | BKL | 14 | 0.028 |
| NV | MEL | 13 | 0.087 |
| SCCKA | AKIEC | 13 | 0.137 |
| BKL | SCCKA | 12 | 0.110 |
| SCCKA | BCC | 12 | 0.126 |
| BKL | AKIEC | 10 | 0.092 |
| AKIEC | SCCKA | 8 | 0.131 |
| BCC | MEL | 7 | 0.014 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 30 | 0.060 |
| BCC | SCCKA | 20 | 0.040 |
| BCC | BKL | 14 | 0.028 |
| SCCKA | AKIEC | 13 | 0.137 |
| BKL | SCCKA | 12 | 0.110 |
| AKIEC | SCCKA | 8 | 0.131 |
| SCCKA | BKL | 6 | 0.063 |
| INF | BCC | 3 | 0.300 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 64.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
