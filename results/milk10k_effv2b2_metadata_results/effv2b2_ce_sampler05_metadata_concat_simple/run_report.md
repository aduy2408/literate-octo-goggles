# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine1_metadata_compact8/effv2b2_ce_sampler05_metadata_concat_simple
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ce
- class_weight: False
- weighted_sampler: True
- augmented_data_dir: None
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7566793893129771
- balanced_accuracy: 0.5502810693274567
- dice_macro: 0.5432292291581526
- f1_macro: 0.5432292291581526
- roc_auc_macro_ovr: 0.9320705107433678
- top2_accuracy: 0.8940839694656488
- top3_accuracy: 0.9446564885496184

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
| AKIEC | 61 | 0.476744 | 0.672131 | 0.954407 | 0.557823 | 0.955238 |
| BCC | 504 | 0.903967 | 0.859127 | 0.915441 | 0.880977 | 0.964074 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.90493 |
| BKL | 109 | 0.580357 | 0.59633 | 0.949947 | 0.588235 | 0.897216 |
| DF | 10 | 0.636364 | 0.7 | 0.996146 | 0.666667 | 0.987091 |
| INF | 10 | 0.5 | 0.4 | 0.996146 | 0.444444 | 0.939114 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.757648 |
| MEL | 90 | 0.643678 | 0.622222 | 0.967641 | 0.632768 | 0.945604 |
| NV | 149 | 0.788462 | 0.825503 | 0.963293 | 0.806557 | 0.980284 |
| SCCKA | 95 | 0.612903 | 0.6 | 0.962225 | 0.606383 | 0.928956 |
| VASC | 9 | 0.666667 | 0.666667 | 0.997113 | 0.666667 | 0.992621 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.757648 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.90493 |
| INF | 10 | 0.5 | 0.4 | 0.996146 | 0.444444 | 0.939114 |
| AKIEC | 61 | 0.476744 | 0.672131 | 0.954407 | 0.557823 | 0.955238 |
| BKL | 109 | 0.580357 | 0.59633 | 0.949947 | 0.588235 | 0.897216 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 86 | 0.0885 |
| BCC | 479 | 0.4239 |
| BEN_OTH | 7 | 0.0108 |
| BKL | 112 | 0.1219 |
| DF | 11 | 0.0106 |
| INF | 8 | 0.0080 |
| MAL_OTH | 0 | 0.0004 |
| MEL | 87 | 0.0869 |
| NV | 156 | 0.1453 |
| SCCKA | 93 | 0.0938 |
| VASC | 9 | 0.0100 |

- mean_confidence: 0.8223088979721069
- median_confidence: 0.9004234075546265
- mean_top1_top2_gap: 0.7034785747528076
- mean_entropy: 0.4928862452507019
- low_confidence_rows: 94

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 21 | 0.233 |
| BCC | AKIEC | 20 | 0.040 |
| BCC | SCCKA | 19 | 0.038 |
| BCC | BKL | 16 | 0.032 |
| BKL | BCC | 16 | 0.147 |
| NV | MEL | 14 | 0.094 |
| SCCKA | AKIEC | 14 | 0.147 |
| SCCKA | BKL | 11 | 0.116 |
| AKIEC | BKL | 10 | 0.164 |
| BKL | SCCKA | 10 | 0.092 |
| SCCKA | BCC | 10 | 0.105 |
| BKL | AKIEC | 9 | 0.083 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 20 | 0.040 |
| BCC | SCCKA | 19 | 0.038 |
| BCC | BKL | 16 | 0.032 |
| SCCKA | AKIEC | 14 | 0.147 |
| SCCKA | BKL | 11 | 0.116 |
| BKL | SCCKA | 10 | 0.092 |
| AKIEC | SCCKA | 5 | 0.082 |
| INF | BCC | 3 | 0.300 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 55.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
