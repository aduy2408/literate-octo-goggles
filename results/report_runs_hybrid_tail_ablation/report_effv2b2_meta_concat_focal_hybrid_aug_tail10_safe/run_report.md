# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_hybrid_tail_ablation/report_effv2b2_meta_concat_focal_hybrid_aug_tail10_safe
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: focal
- class_weight: False
- weighted_sampler: False
- balance_mode: hybrid
- balance_head_ratio: 2.0
- balance_tail_floor: 100
- balance_min_source_count: 20
- augmented_data_dir: /marimo/milk10k_augmented
- dermoscopic_mask_dir: None
- min_dermoscopic_mask_ratio: 0.01
- augmented_classes: ['BEN_OTH', 'DF', 'INF', 'VASC']
- augmented_max_per_class: 10
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.75
- balanced_accuracy: 0.55686011859276
- dice_macro: 0.5437071203388025
- f1_macro: 0.5437071203388025
- roc_auc_macro_ovr: 0.8984603071512176
- top2_accuracy: 0.892175572519084
- top3_accuracy: 0.9446564885496184

## Data Distribution

### Train

- rows: 4232
- real_rows: 4192
- synthetic_rows: 40
- ignore_metadata_rows: 0

| class | count | synthetic |
|---|---:|---:|
| AKIEC | 242 | 0 |
| BCC | 2018 | 0 |
| BEN_OTH | 45 | 10 |
| BKL | 435 | 0 |
| DF | 52 | 10 |
| INF | 50 | 10 |
| MAL_OTH | 7 | 0 |
| MEL | 360 | 0 |
| NV | 597 | 0 |
| SCCKA | 378 | 0 |
| VASC | 48 | 10 |

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

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AKIEC | 61 | 0.485714 | 0.557377 | 0.963526 | 0.519084 | 0.950255 | 0.609003 |
| BCC | 504 | 0.879208 | 0.880952 | 0.887868 | 0.880079 | 0.953818 | 0.698605 |
| BEN_OTH | 9 | 0.222222 | 0.222222 | 0.993263 | 0.222222 | 0.821516 | 0.428232 |
| BKL | 109 | 0.568627 | 0.53211 | 0.953142 | 0.549763 | 0.873875 | 0.66498 |
| DF | 10 | 0.888889 | 0.8 | 0.999037 | 0.842105 | 0.930925 | 0.835157 |
| INF | 10 | 0.277778 | 0.5 | 0.987476 | 0.357143 | 0.964451 | 0.449766 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.583652 |  |
| MEL | 90 | 0.742424 | 0.544444 | 0.982255 | 0.628205 | 0.927975 | 0.739948 |
| NV | 149 | 0.78481 | 0.832215 | 0.96218 | 0.807818 | 0.969131 | 0.755598 |
| SCCKA | 95 | 0.56 | 0.589474 | 0.95383 | 0.574359 | 0.928105 | 0.624056 |
| VASC | 9 | 0.545455 | 0.666667 | 0.995188 | 0.6 | 0.97936 | 0.849777 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.583652 |  |
| BEN_OTH | 9 | 0.222222 | 0.222222 | 0.993263 | 0.222222 | 0.821516 | 0.428232 |
| INF | 10 | 0.277778 | 0.5 | 0.987476 | 0.357143 | 0.964451 | 0.449766 |
| AKIEC | 61 | 0.485714 | 0.557377 | 0.963526 | 0.519084 | 0.950255 | 0.609003 |
| BKL | 109 | 0.568627 | 0.53211 | 0.953142 | 0.549763 | 0.873875 | 0.66498 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 70 | 0.0884 |
| BCC | 505 | 0.3695 |
| BEN_OTH | 9 | 0.0272 |
| BKL | 102 | 0.1414 |
| DF | 9 | 0.0178 |
| INF | 18 | 0.0217 |
| MAL_OTH | 0 | 0.0010 |
| MEL | 66 | 0.0834 |
| NV | 158 | 0.1357 |
| SCCKA | 100 | 0.1021 |
| VASC | 11 | 0.0118 |

- mean_confidence: 0.649728536605835
- median_confidence: 0.6694055795669556
- mean_top1_top2_gap: 0.45700716972351074
- mean_entropy: 0.9989244341850281
- low_confidence_rows: 245

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 23 | 0.256 |
| BKL | BCC | 22 | 0.202 |
| BCC | SCCKA | 17 | 0.034 |
| BKL | SCCKA | 16 | 0.147 |
| BCC | BKL | 15 | 0.030 |
| SCCKA | AKIEC | 15 | 0.158 |
| SCCKA | BCC | 15 | 0.158 |
| BCC | AKIEC | 11 | 0.022 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | BKL | 9 | 0.095 |
| AKIEC | BCC | 8 | 0.131 |
| NV | BKL | 8 | 0.054 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 17 | 0.034 |
| BKL | SCCKA | 16 | 0.147 |
| BCC | BKL | 15 | 0.030 |
| SCCKA | AKIEC | 15 | 0.158 |
| BCC | AKIEC | 11 | 0.022 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | BKL | 9 | 0.095 |
| INF | BCC | 2 | 0.200 |
| INF | BEN_OTH | 2 | 0.200 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 43.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
