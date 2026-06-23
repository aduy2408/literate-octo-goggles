# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine2_no_metadata_compact8/effv2b2_focal_plain_no_metadata_concat_simple
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: focal
- class_weight: False
- weighted_sampler: False
- augmented_data_dir: None
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7662213740458015
- balanced_accuracy: 0.4797525370293502
- dice_macro: 0.5116353853457007
- f1_macro: 0.5116353853457007
- roc_auc_macro_ovr: 0.8848576029564706
- top2_accuracy: 0.898854961832061
- top3_accuracy: 0.9408396946564885

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
| AKIEC | 61 | 0.68 | 0.557377 | 0.983789 | 0.612613 | 0.955902 |
| BCC | 504 | 0.838652 | 0.938492 | 0.832721 | 0.885768 | 0.947712 |
| BEN_OTH | 9 | 0 | 0 | 1 | 0 | 0.805155 |
| BKL | 109 | 0.632911 | 0.458716 | 0.969116 | 0.531915 | 0.863265 |
| DF | 10 | 1 | 0.6 | 1 | 0.75 | 0.961946 |
| INF | 10 | 1 | 0.1 | 1 | 0.181818 | 0.928709 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.457935 |
| MEL | 90 | 0.791667 | 0.422222 | 0.989562 | 0.550725 | 0.925725 |
| NV | 149 | 0.683417 | 0.912752 | 0.929922 | 0.781609 | 0.971124 |
| SCCKA | 95 | 0.634409 | 0.621053 | 0.964323 | 0.62766 | 0.926227 |
| VASC | 9 | 0.75 | 0.666667 | 0.998075 | 0.705882 | 0.989734 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.457935 |
| BEN_OTH | 9 | 0 | 0 | 1 | 0 | 0.805155 |
| INF | 10 | 1 | 0.1 | 1 | 0.181818 | 0.928709 |
| BKL | 109 | 0.632911 | 0.458716 | 0.969116 | 0.531915 | 0.863265 |
| MEL | 90 | 0.791667 | 0.422222 | 0.989562 | 0.550725 | 0.925725 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 50 | 0.0763 |
| BCC | 564 | 0.3917 |
| BEN_OTH | 0 | 0.0110 |
| BKL | 79 | 0.1401 |
| DF | 6 | 0.0118 |
| INF | 1 | 0.0093 |
| MAL_OTH | 0 | 0.0014 |
| MEL | 48 | 0.0702 |
| NV | 199 | 0.1732 |
| SCCKA | 93 | 0.1079 |
| VASC | 8 | 0.0072 |

- mean_confidence: 0.6445417404174805
- median_confidence: 0.6630379557609558
- mean_top1_top2_gap: 0.4591710865497589
- mean_entropy: 1.0489367246627808
- low_confidence_rows: 240

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 40 | 0.444 |
| BKL | BCC | 30 | 0.275 |
| SCCKA | BCC | 18 | 0.189 |
| BCC | SCCKA | 15 | 0.030 |
| BKL | NV | 15 | 0.138 |
| AKIEC | BCC | 12 | 0.197 |
| BKL | SCCKA | 10 | 0.092 |
| SCCKA | AKIEC | 9 | 0.095 |
| AKIEC | SCCKA | 8 | 0.131 |
| INF | BCC | 8 | 0.800 |
| SCCKA | BKL | 8 | 0.084 |
| AKIEC | BKL | 7 | 0.115 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 15 | 0.030 |
| BKL | SCCKA | 10 | 0.092 |
| SCCKA | AKIEC | 9 | 0.095 |
| AKIEC | SCCKA | 8 | 0.131 |
| INF | BCC | 8 | 0.800 |
| SCCKA | BKL | 8 | 0.084 |
| BCC | AKIEC | 5 | 0.010 |
| BCC | BKL | 3 | 0.006 |

## Warnings

- [high] tail_predicted_zero: BEN_OTH has zero predicted rows.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
