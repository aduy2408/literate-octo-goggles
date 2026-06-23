# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/dual_encoder_ablation_runs_v2/run_tau050_sampler_ema
- backbone: efficientnet_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ce
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

- accuracy: 0.767175572519084
- balanced_accuracy: 0.5603745164271472
- dice_macro: 0.5679354963815673
- f1_macro: 0.5679354963815673
- roc_auc_macro_ovr: 0.9050422309968418
- top2_accuracy: 0.875
- top3_accuracy: 0.9303435114503816

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

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AKIEC | 61 | 0.576923 | 0.491803 | 0.97771 | 0.530973 | 0.933297 | 0.665448 |
| BCC | 504 | 0.892057 | 0.869048 | 0.902574 | 0.880402 | 0.955361 | 0.828813 |
| BEN_OTH | 9 | 0.666667 | 0.222222 | 0.999038 | 0.333333 | 0.943963 | 0.356659 |
| BKL | 109 | 0.541284 | 0.541284 | 0.946752 | 0.541284 | 0.857305 | 0.76573 |
| DF | 10 | 0.777778 | 0.7 | 0.998073 | 0.736842 | 0.988536 | 0.764265 |
| INF | 10 | 0.375 | 0.3 | 0.995183 | 0.333333 | 0.918882 | 0.593769 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.549713 |  |
| MEL | 90 | 0.655556 | 0.655556 | 0.967641 | 0.655556 | 0.935827 | 0.871486 |
| NV | 149 | 0.864865 | 0.85906 | 0.977753 | 0.861953 | 0.973341 | 0.868518 |
| SCCKA | 95 | 0.554688 | 0.747368 | 0.940189 | 0.636771 | 0.941481 | 0.791139 |
| VASC | 9 | 0.7 | 0.777778 | 0.997113 | 0.736842 | 0.957759 | 0.908371 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.549713 |  |
| BEN_OTH | 9 | 0.666667 | 0.222222 | 0.999038 | 0.333333 | 0.943963 | 0.356659 |
| INF | 10 | 0.375 | 0.3 | 0.995183 | 0.333333 | 0.918882 | 0.593769 |
| AKIEC | 61 | 0.576923 | 0.491803 | 0.97771 | 0.530973 | 0.933297 | 0.665448 |
| BKL | 109 | 0.541284 | 0.541284 | 0.946752 | 0.541284 | 0.857305 | 0.76573 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 52 | 0.0720 |
| BCC | 491 | 0.4049 |
| BEN_OTH | 3 | 0.0112 |
| BKL | 109 | 0.1173 |
| DF | 9 | 0.0155 |
| INF | 8 | 0.0150 |
| MAL_OTH | 0 | 0.0050 |
| MEL | 90 | 0.0954 |
| NV | 148 | 0.1345 |
| SCCKA | 128 | 0.1156 |
| VASC | 10 | 0.0136 |

- mean_confidence: 0.7744271159172058
- median_confidence: 0.8478093147277832
- mean_top1_top2_gap: 0.6528069972991943
- mean_entropy: 0.708600640296936
- low_confidence_rows: 136

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 25 | 0.050 |
| BCC | SCCKA | 24 | 0.048 |
| BKL | BCC | 21 | 0.193 |
| AKIEC | SCCKA | 18 | 0.295 |
| MEL | NV | 15 | 0.167 |
| BKL | SCCKA | 13 | 0.119 |
| NV | MEL | 11 | 0.074 |
| SCCKA | BCC | 10 | 0.105 |
| BCC | AKIEC | 8 | 0.016 |
| BKL | MEL | 8 | 0.073 |
| MEL | BCC | 8 | 0.089 |
| AKIEC | BKL | 7 | 0.115 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 25 | 0.050 |
| BCC | SCCKA | 24 | 0.048 |
| AKIEC | SCCKA | 18 | 0.295 |
| BKL | SCCKA | 13 | 0.119 |
| BCC | AKIEC | 8 | 0.016 |
| SCCKA | AKIEC | 6 | 0.063 |
| SCCKA | BKL | 5 | 0.053 |
| INF | BCC | 4 | 0.400 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 57.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
