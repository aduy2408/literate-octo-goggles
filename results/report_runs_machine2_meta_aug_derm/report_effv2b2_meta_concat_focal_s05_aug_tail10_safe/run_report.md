# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine2_meta_aug_derm/report_effv2b2_meta_concat_focal_s05_aug_tail10_safe
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
- augmented_classes: ['BEN_OTH', 'DF', 'INF', 'VASC']
- augmented_max_per_class: 10
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7347328244274809
- balanced_accuracy: 0.5483867377154222
- dice_macro: 0.5392420122603512
- f1_macro: 0.5392420122603512
- roc_auc_macro_ovr: 0.9182645763444337
- top2_accuracy: 0.8702290076335878
- top3_accuracy: 0.9408396946564885

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

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| AKIEC | 61 | 0.352459 | 0.704918 | 0.919959 | 0.469945 | 0.937516 |
| BCC | 504 | 0.894967 | 0.811508 | 0.911765 | 0.851197 | 0.945371 |
| BEN_OTH | 9 | 0.5 | 0.222222 | 0.998075 | 0.307692 | 0.868998 |
| BKL | 109 | 0.621053 | 0.541284 | 0.961661 | 0.578431 | 0.872498 |
| DF | 10 | 0.7 | 0.7 | 0.99711 | 0.7 | 0.960405 |
| INF | 10 | 0.333333 | 0.2 | 0.996146 | 0.25 | 0.938728 |
| MAL_OTH | 2 | 0 | 0 | 0.998088 | 0 | 0.797801 |
| MEL | 90 | 0.809524 | 0.566667 | 0.987474 | 0.666667 | 0.936952 |
| NV | 149 | 0.801242 | 0.865772 | 0.964405 | 0.832258 | 0.964853 |
| SCCKA | 95 | 0.521368 | 0.642105 | 0.941238 | 0.575472 | 0.917678 |
| VASC | 9 | 0.636364 | 0.777778 | 0.99615 | 0.7 | 0.960111 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 0.998088 | 0 | 0.797801 |
| INF | 10 | 0.333333 | 0.2 | 0.996146 | 0.25 | 0.938728 |
| BEN_OTH | 9 | 0.5 | 0.222222 | 0.998075 | 0.307692 | 0.868998 |
| AKIEC | 61 | 0.352459 | 0.704918 | 0.919959 | 0.469945 | 0.937516 |
| SCCKA | 95 | 0.521368 | 0.642105 | 0.941238 | 0.575472 | 0.917678 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 122 | 0.1396 |
| BCC | 457 | 0.3274 |
| BEN_OTH | 4 | 0.0121 |
| BKL | 95 | 0.1393 |
| DF | 10 | 0.0153 |
| INF | 6 | 0.0089 |
| MAL_OTH | 2 | 0.0030 |
| MEL | 63 | 0.0768 |
| NV | 161 | 0.1386 |
| SCCKA | 117 | 0.1277 |
| VASC | 11 | 0.0114 |

- mean_confidence: 0.6440799236297607
- median_confidence: 0.6446872353553772
- mean_top1_top2_gap: 0.4498835802078247
- mean_entropy: 0.9863870739936829
- low_confidence_rows: 259

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 42 | 0.083 |
| BCC | SCCKA | 28 | 0.056 |
| MEL | NV | 25 | 0.278 |
| SCCKA | AKIEC | 19 | 0.200 |
| BKL | BCC | 18 | 0.165 |
| BKL | SCCKA | 15 | 0.138 |
| BCC | BKL | 12 | 0.024 |
| BKL | AKIEC | 12 | 0.110 |
| SCCKA | BCC | 10 | 0.105 |
| AKIEC | SCCKA | 9 | 0.148 |
| NV | BKL | 8 | 0.054 |
| AKIEC | BKL | 6 | 0.098 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 42 | 0.083 |
| BCC | SCCKA | 28 | 0.056 |
| SCCKA | AKIEC | 19 | 0.200 |
| BKL | SCCKA | 15 | 0.138 |
| BCC | BKL | 12 | 0.024 |
| AKIEC | SCCKA | 9 | 0.148 |
| INF | BCC | 5 | 0.500 |
| SCCKA | BKL | 4 | 0.042 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 82.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
