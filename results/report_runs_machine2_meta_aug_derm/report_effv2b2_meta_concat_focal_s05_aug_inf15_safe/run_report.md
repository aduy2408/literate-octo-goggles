# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine2_meta_aug_derm/report_effv2b2_meta_concat_focal_s05_aug_inf15_safe
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
- augmented_data_dir: /marimo/milk10k_augmented
- dermoscopic_mask_dir: None
- min_dermoscopic_mask_ratio: 0.01
- augmented_classes: ['INF']
- augmented_max_per_class: 15
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7576335877862596
- balanced_accuracy: 0.5833083109850652
- dice_macro: 0.582813715294477
- f1_macro: 0.582813715294477
- roc_auc_macro_ovr: 0.9321608926585504
- top2_accuracy: 0.8874045801526718
- top3_accuracy: 0.9456106870229007

## Data Distribution

### Train

- rows: 4207
- real_rows: 4192
- synthetic_rows: 15
- ignore_metadata_rows: 0

| class | count | synthetic |
|---|---:|---:|
| AKIEC | 242 | 0 |
| BCC | 2018 | 0 |
| BEN_OTH | 35 | 0 |
| BKL | 435 | 0 |
| DF | 42 | 0 |
| INF | 55 | 15 |
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
| AKIEC | 61 | 0.455556 | 0.672131 | 0.950355 | 0.543046 | 0.952564 |
| BCC | 504 | 0.91757 | 0.839286 | 0.930147 | 0.876684 | 0.956791 |
| BEN_OTH | 9 | 0.5 | 0.222222 | 0.998075 | 0.307692 | 0.876377 |
| BKL | 109 | 0.598131 | 0.587156 | 0.954207 | 0.592593 | 0.872419 |
| DF | 10 | 0.888889 | 0.8 | 0.999037 | 0.842105 | 0.981214 |
| INF | 10 | 0.444444 | 0.4 | 0.995183 | 0.421053 | 0.935453 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.851816 |
| MEL | 90 | 0.679487 | 0.588889 | 0.973904 | 0.630952 | 0.946764 |
| NV | 149 | 0.777108 | 0.865772 | 0.958843 | 0.819048 | 0.972221 |
| SCCKA | 95 | 0.547826 | 0.663158 | 0.945435 | 0.6 | 0.92847 |
| VASC | 9 | 0.777778 | 0.777778 | 0.998075 | 0.777778 | 0.979681 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.851816 |
| BEN_OTH | 9 | 0.5 | 0.222222 | 0.998075 | 0.307692 | 0.876377 |
| INF | 10 | 0.444444 | 0.4 | 0.995183 | 0.421053 | 0.935453 |
| AKIEC | 61 | 0.455556 | 0.672131 | 0.950355 | 0.543046 | 0.952564 |
| BKL | 109 | 0.598131 | 0.587156 | 0.954207 | 0.592593 | 0.872419 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 90 | 0.1099 |
| BCC | 461 | 0.3464 |
| BEN_OTH | 4 | 0.0093 |
| BKL | 107 | 0.1483 |
| DF | 9 | 0.0154 |
| INF | 9 | 0.0104 |
| MAL_OTH | 0 | 0.0011 |
| MEL | 78 | 0.0923 |
| NV | 166 | 0.1435 |
| SCCKA | 115 | 0.1144 |
| VASC | 9 | 0.0090 |

- mean_confidence: 0.692762017250061
- median_confidence: 0.7186448574066162
- mean_top1_top2_gap: 0.5036474466323853
- mean_entropy: 0.838656485080719
- low_confidence_rows: 188

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 26 | 0.052 |
| MEL | NV | 26 | 0.289 |
| BCC | AKIEC | 24 | 0.048 |
| BCC | BKL | 18 | 0.036 |
| BKL | BCC | 16 | 0.147 |
| BKL | SCCKA | 13 | 0.119 |
| SCCKA | AKIEC | 13 | 0.137 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | BKL | 10 | 0.105 |
| BKL | AKIEC | 9 | 0.083 |
| BCC | MEL | 8 | 0.016 |
| NV | MEL | 8 | 0.054 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 26 | 0.052 |
| BCC | AKIEC | 24 | 0.048 |
| BCC | BKL | 18 | 0.036 |
| BKL | SCCKA | 13 | 0.119 |
| SCCKA | AKIEC | 13 | 0.137 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | BKL | 10 | 0.105 |
| INF | BCC | 2 | 0.200 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 68.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
