# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_adaptive_gate_balanced_1point5head_pretrained
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: adaptive_gate
- loss: ce
- class_weight: False
- weighted_sampler: False
- balance_mode: hybrid
- balance_head_ratio: 1.5
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

- accuracy: 0.7690839694656488
- balanced_accuracy: 0.5344786369904337
- dice_macro: 0.5615961013685729
- f1_macro: 0.5615961013685729
- roc_auc_macro_ovr: 0.913764773875793
- top2_accuracy: 0.8959923664122137
- top3_accuracy: 0.9427480916030534

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
| AKIEC | 61 | 0.521739 | 0.590164 | 0.966565 | 0.553846 | 0.950404 |
| BCC | 504 | 0.891089 | 0.892857 | 0.898897 | 0.891972 | 0.962871 |
| BEN_OTH | 9 | 0.5 | 0.111111 | 0.999038 | 0.181818 | 0.835953 |
| BKL | 109 | 0.555556 | 0.59633 | 0.944622 | 0.575221 | 0.894666 |
| DF | 10 | 0.777778 | 0.7 | 0.998073 | 0.736842 | 0.973218 |
| INF | 10 | 0.6 | 0.3 | 0.998073 | 0.4 | 0.916474 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.706023 |
| MEL | 90 | 0.738462 | 0.533333 | 0.982255 | 0.619355 | 0.920761 |
| NV | 149 | 0.752809 | 0.899329 | 0.951057 | 0.819572 | 0.975633 |
| SCCKA | 95 | 0.608696 | 0.589474 | 0.962225 | 0.59893 | 0.93006 |
| VASC | 9 | 1 | 0.666667 | 1 | 0.8 | 0.985349 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.706023 |
| BEN_OTH | 9 | 0.5 | 0.111111 | 0.999038 | 0.181818 | 0.835953 |
| INF | 10 | 0.6 | 0.3 | 0.998073 | 0.4 | 0.916474 |
| AKIEC | 61 | 0.521739 | 0.590164 | 0.966565 | 0.553846 | 0.950404 |
| BKL | 109 | 0.555556 | 0.59633 | 0.944622 | 0.575221 | 0.894666 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 69 | 0.0662 |
| BCC | 505 | 0.4666 |
| BEN_OTH | 2 | 0.0054 |
| BKL | 117 | 0.1187 |
| DF | 9 | 0.0095 |
| INF | 5 | 0.0067 |
| MAL_OTH | 0 | 0.0003 |
| MEL | 65 | 0.0664 |
| NV | 178 | 0.1671 |
| SCCKA | 92 | 0.0869 |
| VASC | 6 | 0.0062 |

- mean_confidence: 0.8977380990982056
- median_confidence: 0.9862706661224365
- mean_top1_top2_gap: 0.8243158459663391
- mean_entropy: 0.2817680239677429
- low_confidence_rows: 45

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 28 | 0.311 |
| BKL | BCC | 18 | 0.165 |
| BCC | SCCKA | 17 | 0.034 |
| BCC | AKIEC | 15 | 0.030 |
| SCCKA | BKL | 14 | 0.147 |
| SCCKA | BCC | 13 | 0.137 |
| AKIEC | BKL | 12 | 0.197 |
| SCCKA | AKIEC | 11 | 0.116 |
| BCC | BKL | 10 | 0.020 |
| BKL | SCCKA | 10 | 0.092 |
| MEL | BCC | 7 | 0.078 |
| AKIEC | BCC | 6 | 0.098 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 17 | 0.034 |
| BCC | AKIEC | 15 | 0.030 |
| SCCKA | BKL | 14 | 0.147 |
| SCCKA | AKIEC | 11 | 0.116 |
| BCC | BKL | 10 | 0.020 |
| BKL | SCCKA | 10 | 0.092 |
| AKIEC | SCCKA | 6 | 0.098 |
| INF | BCC | 3 | 0.300 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 42.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
