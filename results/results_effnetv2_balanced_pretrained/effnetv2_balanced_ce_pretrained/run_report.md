# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_balanced_ce_pretrained
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
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

- accuracy: 0.7595419847328244
- balanced_accuracy: 0.5748741533929417
- dice_macro: 0.5446185297901888
- f1_macro: 0.5446185297901888
- roc_auc_macro_ovr: 0.8956965460579469
- top2_accuracy: 0.8912213740458015
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
| AKIEC | 61 | 0.573529 | 0.639344 | 0.970618 | 0.604651 | 0.956932 |
| BCC | 504 | 0.900826 | 0.865079 | 0.911765 | 0.882591 | 0.964213 |
| BEN_OTH | 9 | 0.375 | 0.333333 | 0.995188 | 0.352941 | 0.817132 |
| BKL | 109 | 0.552632 | 0.577982 | 0.945687 | 0.565022 | 0.907251 |
| DF | 10 | 0.615385 | 0.8 | 0.995183 | 0.695652 | 0.979287 |
| INF | 10 | 0.166667 | 0.3 | 0.985549 | 0.214286 | 0.923314 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.486616 |
| MEL | 90 | 0.723077 | 0.522222 | 0.981211 | 0.606452 | 0.924414 |
| NV | 149 | 0.732955 | 0.865772 | 0.94772 | 0.793846 | 0.974155 |
| SCCKA | 95 | 0.693182 | 0.642105 | 0.971668 | 0.666667 | 0.933893 |
| VASC | 9 | 0.5 | 0.777778 | 0.993263 | 0.608696 | 0.985456 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.486616 |
| INF | 10 | 0.166667 | 0.3 | 0.985549 | 0.214286 | 0.923314 |
| BEN_OTH | 9 | 0.375 | 0.333333 | 0.995188 | 0.352941 | 0.817132 |
| BKL | 109 | 0.552632 | 0.577982 | 0.945687 | 0.565022 | 0.907251 |
| AKIEC | 61 | 0.573529 | 0.639344 | 0.970618 | 0.604651 | 0.956932 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 68 | 0.0690 |
| BCC | 484 | 0.4432 |
| BEN_OTH | 8 | 0.0099 |
| BKL | 114 | 0.1113 |
| DF | 13 | 0.0148 |
| INF | 18 | 0.0208 |
| MAL_OTH | 0 | 0.0006 |
| MEL | 65 | 0.0698 |
| NV | 176 | 0.1643 |
| SCCKA | 88 | 0.0834 |
| VASC | 14 | 0.0130 |

- mean_confidence: 0.878604531288147
- median_confidence: 0.978050947189331
- mean_top1_top2_gap: 0.7977679967880249
- mean_entropy: 0.3417147696018219
- low_confidence_rows: 74

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 29 | 0.322 |
| BKL | BCC | 20 | 0.183 |
| BCC | BKL | 17 | 0.034 |
| BCC | AKIEC | 13 | 0.026 |
| SCCKA | BCC | 12 | 0.126 |
| BCC | INF | 10 | 0.020 |
| BCC | SCCKA | 10 | 0.020 |
| NV | MEL | 10 | 0.067 |
| SCCKA | BKL | 10 | 0.105 |
| AKIEC | SCCKA | 9 | 0.148 |
| SCCKA | AKIEC | 9 | 0.095 |
| AKIEC | BKL | 8 | 0.131 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 17 | 0.034 |
| BCC | AKIEC | 13 | 0.026 |
| BCC | SCCKA | 10 | 0.020 |
| SCCKA | BKL | 10 | 0.105 |
| AKIEC | SCCKA | 9 | 0.148 |
| SCCKA | AKIEC | 9 | 0.095 |
| BKL | SCCKA | 8 | 0.073 |
| INF | BCC | 4 | 0.400 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 40.
- [high] tail_precision_low: INF recall=0.300 but precision=0.167.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
