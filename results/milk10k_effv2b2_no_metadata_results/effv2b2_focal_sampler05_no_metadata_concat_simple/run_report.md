# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine2_no_metadata_compact8/effv2b2_focal_sampler05_no_metadata_concat_simple
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: focal
- class_weight: False
- weighted_sampler: True
- augmented_data_dir: None
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.75
- balanced_accuracy: 0.5371988373320126
- dice_macro: 0.5468194856063177
- f1_macro: 0.5468194856063177
- roc_auc_macro_ovr: 0.9052307704293924
- top2_accuracy: 0.8931297709923665
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
| AKIEC | 61 | 0.435897 | 0.836066 | 0.933131 | 0.573034 | 0.95072 |
| BCC | 504 | 0.91258 | 0.849206 | 0.924632 | 0.879753 | 0.95586 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.812106 |
| BKL | 109 | 0.582524 | 0.550459 | 0.954207 | 0.566038 | 0.862473 |
| DF | 10 | 1 | 0.6 | 1 | 0.75 | 0.985164 |
| INF | 10 | 0.428571 | 0.3 | 0.996146 | 0.352941 | 0.91079 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.661568 |
| MEL | 90 | 0.59375 | 0.633333 | 0.95929 | 0.612903 | 0.937625 |
| NV | 149 | 0.759259 | 0.825503 | 0.956618 | 0.790997 | 0.972617 |
| SCCKA | 95 | 0.671053 | 0.536842 | 0.973767 | 0.596491 | 0.921235 |
| VASC | 9 | 0.857143 | 0.666667 | 0.999038 | 0.75 | 0.987381 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.661568 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.812106 |
| INF | 10 | 0.428571 | 0.3 | 0.996146 | 0.352941 | 0.91079 |
| BKL | 109 | 0.582524 | 0.550459 | 0.954207 | 0.566038 | 0.862473 |
| AKIEC | 61 | 0.435897 | 0.836066 | 0.933131 | 0.573034 | 0.95072 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 117 | 0.1122 |
| BCC | 469 | 0.3700 |
| BEN_OTH | 5 | 0.0102 |
| BKL | 103 | 0.1422 |
| DF | 6 | 0.0081 |
| INF | 7 | 0.0098 |
| MAL_OTH | 0 | 0.0003 |
| MEL | 96 | 0.1124 |
| NV | 162 | 0.1398 |
| SCCKA | 76 | 0.0869 |
| VASC | 7 | 0.0082 |

- mean_confidence: 0.7224571704864502
- median_confidence: 0.739928126335144
- mean_top1_top2_gap: 0.5319026708602905
- mean_entropy: 0.7453571557998657
- low_confidence_rows: 113

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| SCCKA | AKIEC | 25 | 0.263 |
| BCC | AKIEC | 24 | 0.048 |
| MEL | NV | 23 | 0.256 |
| BCC | BKL | 19 | 0.038 |
| NV | MEL | 17 | 0.114 |
| BKL | BCC | 15 | 0.138 |
| BCC | SCCKA | 14 | 0.028 |
| BKL | AKIEC | 14 | 0.128 |
| SCCKA | BKL | 10 | 0.105 |
| BCC | MEL | 9 | 0.018 |
| BKL | NV | 7 | 0.064 |
| BKL | SCCKA | 7 | 0.064 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| SCCKA | AKIEC | 25 | 0.263 |
| BCC | AKIEC | 24 | 0.048 |
| BCC | BKL | 19 | 0.038 |
| BCC | SCCKA | 14 | 0.028 |
| SCCKA | BKL | 10 | 0.105 |
| BKL | SCCKA | 7 | 0.064 |
| INF | BCC | 4 | 0.400 |
| AKIEC | SCCKA | 2 | 0.033 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 57.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
