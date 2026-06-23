# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine2_no_metadata_compact8/effv2b2_ce_classweight_no_metadata_concat_simple
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

- accuracy: 0.7223282442748091
- balanced_accuracy: 0.534249636046005
- dice_macro: 0.5175290507581048
- f1_macro: 0.5175290507581048
- roc_auc_macro_ovr: 0.8863116901367306
- top2_accuracy: 0.8740458015267175
- top3_accuracy: 0.9293893129770993

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
| AKIEC | 61 | 0.492754 | 0.557377 | 0.964539 | 0.523077 | 0.949856 |
| BCC | 504 | 0.88326 | 0.795635 | 0.902574 | 0.837161 | 0.941658 |
| BEN_OTH | 9 | 0.333333 | 0.111111 | 0.998075 | 0.166667 | 0.791038 |
| BKL | 109 | 0.626506 | 0.477064 | 0.966986 | 0.541667 | 0.850651 |
| DF | 10 | 0.545455 | 0.6 | 0.995183 | 0.571429 | 0.977746 |
| INF | 10 | 0.333333 | 0.4 | 0.992293 | 0.363636 | 0.906551 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.540153 |
| MEL | 90 | 0.732143 | 0.455556 | 0.984342 | 0.561644 | 0.904547 |
| NV | 149 | 0.697436 | 0.912752 | 0.934372 | 0.790698 | 0.971116 |
| SCCKA | 95 | 0.483871 | 0.789474 | 0.916055 | 0.6 | 0.926272 |
| VASC | 9 | 0.7 | 0.777778 | 0.997113 | 0.736842 | 0.989841 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.540153 |
| BEN_OTH | 9 | 0.333333 | 0.111111 | 0.998075 | 0.166667 | 0.791038 |
| INF | 10 | 0.333333 | 0.4 | 0.992293 | 0.363636 | 0.906551 |
| AKIEC | 61 | 0.492754 | 0.557377 | 0.964539 | 0.523077 | 0.949856 |
| BKL | 109 | 0.626506 | 0.477064 | 0.966986 | 0.541667 | 0.850651 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 69 | 0.0811 |
| BCC | 454 | 0.3362 |
| BEN_OTH | 3 | 0.0155 |
| BKL | 83 | 0.1184 |
| DF | 11 | 0.0164 |
| INF | 12 | 0.0178 |
| MAL_OTH | 0 | 0.0025 |
| MEL | 56 | 0.0634 |
| NV | 195 | 0.1812 |
| SCCKA | 155 | 0.1566 |
| VASC | 10 | 0.0110 |

- mean_confidence: 0.7266356348991394
- median_confidence: 0.7641853094100952
- mean_top1_top2_gap: 0.586956799030304
- mean_entropy: 0.828264057636261
- low_confidence_rows: 195

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 47 | 0.093 |
| MEL | NV | 37 | 0.411 |
| BCC | AKIEC | 22 | 0.044 |
| BKL | BCC | 19 | 0.174 |
| AKIEC | SCCKA | 15 | 0.246 |
| BKL | SCCKA | 15 | 0.138 |
| BCC | BKL | 13 | 0.026 |
| BKL | NV | 13 | 0.119 |
| SCCKA | BCC | 10 | 0.105 |
| AKIEC | BCC | 7 | 0.115 |
| BCC | MEL | 7 | 0.014 |
| SCCKA | AKIEC | 7 | 0.074 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 47 | 0.093 |
| BCC | AKIEC | 22 | 0.044 |
| AKIEC | SCCKA | 15 | 0.246 |
| BKL | SCCKA | 15 | 0.138 |
| BCC | BKL | 13 | 0.026 |
| SCCKA | AKIEC | 7 | 0.074 |
| INF | BCC | 4 | 0.400 |
| SCCKA | BKL | 3 | 0.032 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 82.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
