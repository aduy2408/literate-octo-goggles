# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_adaptive_gate_balanced_pretrained
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: adaptive_gate
- loss: ce
- class_weight: False
- weighted_sampler: False
- balance_mode: hybrid
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

- accuracy: 0.7662213740458015
- balanced_accuracy: 0.5917921748031305
- dice_macro: 0.5766656189500702
- f1_macro: 0.5766656189500702
- roc_auc_macro_ovr: 0.8965699964951468
- top2_accuracy: 0.8826335877862596
- top3_accuracy: 0.9360687022900763

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
| AKIEC | 61 | 0.571429 | 0.590164 | 0.972644 | 0.580645 | 0.948677 |
| BCC | 504 | 0.88978 | 0.880952 | 0.898897 | 0.885344 | 0.955554 |
| BEN_OTH | 9 | 0.428571 | 0.333333 | 0.99615 | 0.375 | 0.820768 |
| BKL | 109 | 0.666667 | 0.513761 | 0.970181 | 0.580311 | 0.874549 |
| DF | 10 | 0.8 | 0.8 | 0.998073 | 0.8 | 0.980925 |
| INF | 10 | 0.333333 | 0.4 | 0.992293 | 0.363636 | 0.946243 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.503824 |
| MEL | 90 | 0.725806 | 0.5 | 0.982255 | 0.592105 | 0.923776 |
| NV | 149 | 0.732955 | 0.865772 | 0.94772 | 0.793846 | 0.972244 |
| SCCKA | 95 | 0.57377 | 0.736842 | 0.945435 | 0.645161 | 0.939239 |
| VASC | 9 | 0.615385 | 0.888889 | 0.995188 | 0.727273 | 0.996471 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.503824 |
| INF | 10 | 0.333333 | 0.4 | 0.992293 | 0.363636 | 0.946243 |
| BEN_OTH | 9 | 0.428571 | 0.333333 | 0.99615 | 0.375 | 0.820768 |
| BKL | 109 | 0.666667 | 0.513761 | 0.970181 | 0.580311 | 0.874549 |
| AKIEC | 61 | 0.571429 | 0.590164 | 0.972644 | 0.580645 | 0.948677 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 63 | 0.0674 |
| BCC | 499 | 0.4334 |
| BEN_OTH | 7 | 0.0137 |
| BKL | 84 | 0.1038 |
| DF | 10 | 0.0110 |
| INF | 12 | 0.0170 |
| MAL_OTH | 0 | 0.0005 |
| MEL | 62 | 0.0667 |
| NV | 176 | 0.1652 |
| SCCKA | 122 | 0.1078 |
| VASC | 13 | 0.0134 |

- mean_confidence: 0.8265278935432434
- median_confidence: 0.9157643914222717
- mean_top1_top2_gap: 0.723779559135437
- mean_entropy: 0.5149667859077454
- low_confidence_rows: 102

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 31 | 0.344 |
| BKL | BCC | 24 | 0.220 |
| BCC | SCCKA | 23 | 0.046 |
| BKL | SCCKA | 15 | 0.138 |
| AKIEC | SCCKA | 12 | 0.197 |
| SCCKA | AKIEC | 11 | 0.116 |
| SCCKA | BCC | 11 | 0.116 |
| BCC | AKIEC | 10 | 0.020 |
| BCC | BKL | 9 | 0.018 |
| BKL | NV | 8 | 0.073 |
| NV | MEL | 8 | 0.054 |
| AKIEC | BKL | 7 | 0.115 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 23 | 0.046 |
| BKL | SCCKA | 15 | 0.138 |
| AKIEC | SCCKA | 12 | 0.197 |
| SCCKA | AKIEC | 11 | 0.116 |
| BCC | AKIEC | 10 | 0.020 |
| BCC | BKL | 9 | 0.018 |
| INF | BCC | 4 | 0.400 |
| SCCKA | BKL | 3 | 0.032 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 42.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
