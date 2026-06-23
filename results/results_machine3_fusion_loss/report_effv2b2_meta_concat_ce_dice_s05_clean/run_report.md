# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine3_fusion_loss/report_effv2b2_meta_concat_ce_dice_s05_clean
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ce_dice
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

- accuracy: 0.7309160305343512
- balanced_accuracy: 0.5784165310325601
- dice_macro: 0.5571035137886123
- f1_macro: 0.5571035137886123
- roc_auc_macro_ovr: 0.9276909983125414
- top2_accuracy: 0.8797709923664122
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
| AKIEC | 61 | 0.419048 | 0.721311 | 0.938197 | 0.53012 | 0.939542 |
| BCC | 504 | 0.909091 | 0.793651 | 0.926471 | 0.847458 | 0.954591 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.859587 |
| BKL | 109 | 0.565217 | 0.59633 | 0.946752 | 0.580357 | 0.85834 |
| DF | 10 | 0.888889 | 0.8 | 0.999037 | 0.842105 | 0.99104 |
| INF | 10 | 0.416667 | 0.5 | 0.993256 | 0.454545 | 0.957611 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.846558 |
| MEL | 90 | 0.611111 | 0.611111 | 0.963466 | 0.611111 | 0.938379 |
| NV | 149 | 0.798658 | 0.798658 | 0.96663 | 0.798658 | 0.974662 |
| SCCKA | 95 | 0.558559 | 0.652632 | 0.948583 | 0.601942 | 0.923002 |
| VASC | 9 | 0.7 | 0.777778 | 0.997113 | 0.736842 | 0.961288 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.846558 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.859587 |
| INF | 10 | 0.416667 | 0.5 | 0.993256 | 0.454545 | 0.957611 |
| AKIEC | 61 | 0.419048 | 0.721311 | 0.938197 | 0.53012 | 0.939542 |
| BKL | 109 | 0.565217 | 0.59633 | 0.946752 | 0.580357 | 0.85834 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 105 | 0.1096 |
| BCC | 440 | 0.3721 |
| BEN_OTH | 7 | 0.0114 |
| BKL | 115 | 0.1328 |
| DF | 9 | 0.0109 |
| INF | 12 | 0.0100 |
| MAL_OTH | 0 | 0.0007 |
| MEL | 90 | 0.0938 |
| NV | 149 | 0.1391 |
| SCCKA | 111 | 0.1086 |
| VASC | 10 | 0.0110 |

- mean_confidence: 0.7893943190574646
- median_confidence: 0.8535981178283691
- mean_top1_top2_gap: 0.6526934504508972
- mean_entropy: 0.5772278904914856
- low_confidence_rows: 120

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 33 | 0.065 |
| BCC | BKL | 30 | 0.060 |
| BCC | SCCKA | 23 | 0.046 |
| MEL | NV | 20 | 0.222 |
| NV | MEL | 18 | 0.121 |
| SCCKA | AKIEC | 17 | 0.179 |
| BKL | BCC | 16 | 0.147 |
| BKL | SCCKA | 14 | 0.128 |
| AKIEC | SCCKA | 10 | 0.164 |
| BKL | AKIEC | 8 | 0.073 |
| SCCKA | BKL | 8 | 0.084 |
| BCC | MEL | 7 | 0.014 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 33 | 0.065 |
| BCC | BKL | 30 | 0.060 |
| BCC | SCCKA | 23 | 0.046 |
| SCCKA | AKIEC | 17 | 0.179 |
| BKL | SCCKA | 14 | 0.128 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | BKL | 8 | 0.084 |
| INF | BCC | 2 | 0.200 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 86.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
