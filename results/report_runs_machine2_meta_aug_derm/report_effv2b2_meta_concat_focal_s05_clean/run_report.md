# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine2_meta_aug_derm/report_effv2b2_meta_concat_focal_s05_clean
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

- accuracy: 0.6812977099236641
- balanced_accuracy: 0.5614229254079605
- dice_macro: 0.5429593101384124
- f1_macro: 0.5429593101384124
- roc_auc_macro_ovr: 0.919674372578653
- top2_accuracy: 0.8568702290076335
- top3_accuracy: 0.9341603053435115

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
| AKIEC | 61 | 0.34058 | 0.770492 | 0.907801 | 0.472362 | 0.9371 |
| BCC | 504 | 0.924933 | 0.684524 | 0.948529 | 0.786773 | 0.945221 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.86087 |
| BKL | 109 | 0.550847 | 0.59633 | 0.943557 | 0.572687 | 0.843871 |
| DF | 10 | 1 | 0.8 | 1 | 0.888889 | 0.964355 |
| INF | 10 | 0.555556 | 0.5 | 0.996146 | 0.526316 | 0.949807 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.831262 |
| MEL | 90 | 0.690141 | 0.544444 | 0.977035 | 0.608696 | 0.932301 |
| NV | 149 | 0.766871 | 0.838926 | 0.957731 | 0.801282 | 0.973983 |
| SCCKA | 95 | 0.409091 | 0.663158 | 0.904512 | 0.506024 | 0.902458 |
| VASC | 9 | 0.666667 | 0.666667 | 0.997113 | 0.666667 | 0.97519 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.831262 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.86087 |
| AKIEC | 61 | 0.34058 | 0.770492 | 0.907801 | 0.472362 | 0.9371 |
| SCCKA | 95 | 0.409091 | 0.663158 | 0.904512 | 0.506024 | 0.902458 |
| INF | 10 | 0.555556 | 0.5 | 0.996146 | 0.526316 | 0.949807 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 138 | 0.1481 |
| BCC | 373 | 0.2691 |
| BEN_OTH | 5 | 0.0175 |
| BKL | 118 | 0.1484 |
| DF | 8 | 0.0136 |
| INF | 9 | 0.0123 |
| MAL_OTH | 0 | 0.0024 |
| MEL | 71 | 0.0820 |
| NV | 163 | 0.1442 |
| SCCKA | 154 | 0.1539 |
| VASC | 9 | 0.0085 |

- mean_confidence: 0.6205829977989197
- median_confidence: 0.6225548982620239
- mean_top1_top2_gap: 0.4088054299354553
- mean_entropy: 1.0265731811523438
- low_confidence_rows: 301

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 60 | 0.119 |
| BCC | AKIEC | 55 | 0.109 |
| BCC | BKL | 29 | 0.058 |
| MEL | NV | 26 | 0.289 |
| SCCKA | AKIEC | 21 | 0.221 |
| BKL | SCCKA | 15 | 0.138 |
| BKL | BCC | 12 | 0.110 |
| BKL | AKIEC | 11 | 0.101 |
| AKIEC | SCCKA | 10 | 0.164 |
| NV | MEL | 10 | 0.067 |
| BCC | MEL | 8 | 0.016 |
| NV | BKL | 8 | 0.054 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 60 | 0.119 |
| BCC | AKIEC | 55 | 0.109 |
| BCC | BKL | 29 | 0.058 |
| SCCKA | AKIEC | 21 | 0.221 |
| BKL | SCCKA | 15 | 0.138 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | BKL | 6 | 0.063 |
| INF | BCC | 2 | 0.200 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 144.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
