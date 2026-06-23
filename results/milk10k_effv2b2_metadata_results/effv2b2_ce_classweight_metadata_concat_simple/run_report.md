# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine1_metadata_compact8/effv2b2_ce_classweight_metadata_concat_simple
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ce
- class_weight: True
- weighted_sampler: False
- augmented_data_dir: None
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7509541984732825
- balanced_accuracy: 0.5673314317856482
- dice_macro: 0.5500810041042202
- f1_macro: 0.5500810041042202
- roc_auc_macro_ovr: 0.9045530233418017
- top2_accuracy: 0.8883587786259542
- top3_accuracy: 0.9379770992366412

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
| AKIEC | 61 | 0.494624 | 0.754098 | 0.952381 | 0.597403 | 0.954906 |
| BCC | 504 | 0.912854 | 0.831349 | 0.926471 | 0.870197 | 0.957425 |
| BEN_OTH | 9 | 0.4 | 0.222222 | 0.997113 | 0.285714 | 0.780344 |
| BKL | 109 | 0.533898 | 0.577982 | 0.941427 | 0.555066 | 0.87708 |
| DF | 10 | 0.7 | 0.7 | 0.99711 | 0.7 | 0.976493 |
| INF | 10 | 0.25 | 0.3 | 0.991329 | 0.272727 | 0.953276 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.605163 |
| MEL | 90 | 0.701299 | 0.6 | 0.975992 | 0.646707 | 0.944491 |
| NV | 149 | 0.773006 | 0.845638 | 0.958843 | 0.807692 | 0.978627 |
| SCCKA | 95 | 0.6 | 0.631579 | 0.958027 | 0.615385 | 0.92752 |
| VASC | 9 | 0.636364 | 0.777778 | 0.99615 | 0.7 | 0.99476 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.605163 |
| INF | 10 | 0.25 | 0.3 | 0.991329 | 0.272727 | 0.953276 |
| BEN_OTH | 9 | 0.4 | 0.222222 | 0.997113 | 0.285714 | 0.780344 |
| BKL | 109 | 0.533898 | 0.577982 | 0.941427 | 0.555066 | 0.87708 |
| AKIEC | 61 | 0.494624 | 0.754098 | 0.952381 | 0.597403 | 0.954906 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 93 | 0.0972 |
| BCC | 459 | 0.3765 |
| BEN_OTH | 5 | 0.0131 |
| BKL | 118 | 0.1344 |
| DF | 10 | 0.0127 |
| INF | 12 | 0.0131 |
| MAL_OTH | 0 | 0.0023 |
| MEL | 77 | 0.0764 |
| NV | 163 | 0.1585 |
| SCCKA | 100 | 0.1057 |
| VASC | 11 | 0.0100 |

- mean_confidence: 0.7856343984603882
- median_confidence: 0.8496851921081543
- mean_top1_top2_gap: 0.6528703570365906
- mean_entropy: 0.6179932951927185
- low_confidence_rows: 135

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 25 | 0.050 |
| MEL | NV | 24 | 0.267 |
| BCC | AKIEC | 23 | 0.046 |
| BCC | SCCKA | 21 | 0.042 |
| BKL | BCC | 16 | 0.147 |
| SCCKA | AKIEC | 14 | 0.147 |
| BKL | SCCKA | 12 | 0.110 |
| NV | MEL | 10 | 0.067 |
| SCCKA | BKL | 10 | 0.105 |
| BKL | AKIEC | 9 | 0.083 |
| SCCKA | BCC | 9 | 0.095 |
| AKIEC | BKL | 6 | 0.098 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 25 | 0.050 |
| BCC | AKIEC | 23 | 0.046 |
| BCC | SCCKA | 21 | 0.042 |
| SCCKA | AKIEC | 14 | 0.147 |
| BKL | SCCKA | 12 | 0.110 |
| SCCKA | BKL | 10 | 0.105 |
| AKIEC | SCCKA | 6 | 0.098 |
| INF | BCC | 3 | 0.300 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 69.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
