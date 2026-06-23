# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/single_encoder_split_runs/singleenc_pool_nometa_clean
- backbone: efficientnet_b2
- metadata_fusion: concat
- image_fusion: shared_encoder_pool
- loss: ce
- class_weight: False
- weighted_sampler: False
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

- accuracy: 0.691793893129771
- balanced_accuracy: 0.36780841302097095
- dice_macro: 0.3930564561281751
- f1_macro: 0.3930564561281751
- roc_auc_macro_ovr: 0.8796304232453952
- top2_accuracy: 0.8473282442748091
- top3_accuracy: 0.9227099236641222

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
| AKIEC | 61 | 0.545455 | 0.196721 | 0.989868 | 0.289157 | 0.909014 | 0.627414 |
| BCC | 504 | 0.785235 | 0.928571 | 0.764706 | 0.850909 | 0.926343 | 0.869658 |
| BEN_OTH | 9 | 1 | 0.111111 | 1 | 0.2 | 0.794888 | 0.510338 |
| BKL | 109 | 0.395161 | 0.449541 | 0.920128 | 0.420601 | 0.801741 | 0.744372 |
| DF | 10 | 0.333333 | 0.1 | 0.998073 | 0.153846 | 0.938343 | 0.451268 |
| INF | 10 | 0 | 0 | 0.998073 | 0 | 0.925145 |  |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.724187 |  |
| MEL | 90 | 0.561404 | 0.355556 | 0.973904 | 0.435374 | 0.890524 | 0.691032 |
| NV | 149 | 0.688312 | 0.711409 | 0.946607 | 0.69967 | 0.954192 | 0.80329 |
| SCCKA | 95 | 0.617284 | 0.526316 | 0.967471 | 0.568182 | 0.912509 | 0.818872 |
| VASC | 9 | 0.75 | 0.666667 | 0.998075 | 0.705882 | 0.899048 | 0.878905 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.724187 |  |
| INF | 10 | 0 | 0 | 0.998073 | 0 | 0.925145 |  |
| DF | 10 | 0.333333 | 0.1 | 0.998073 | 0.153846 | 0.938343 | 0.451268 |
| BEN_OTH | 9 | 1 | 0.111111 | 1 | 0.2 | 0.794888 | 0.510338 |
| AKIEC | 61 | 0.545455 | 0.196721 | 0.989868 | 0.289157 | 0.909014 | 0.627414 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 22 | 0.0390 |
| BCC | 596 | 0.5021 |
| BEN_OTH | 1 | 0.0065 |
| BKL | 124 | 0.1290 |
| DF | 3 | 0.0087 |
| INF | 2 | 0.0106 |
| MAL_OTH | 0 | 0.0009 |
| MEL | 57 | 0.0683 |
| NV | 154 | 0.1380 |
| SCCKA | 81 | 0.0864 |
| VASC | 8 | 0.0105 |

- mean_confidence: 0.7804357409477234
- median_confidence: 0.8398974537849426
- mean_top1_top2_gap: 0.6543256640434265
- mean_entropy: 0.6668052673339844
- low_confidence_rows: 134

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 42 | 0.385 |
| SCCKA | BCC | 31 | 0.326 |
| MEL | NV | 29 | 0.322 |
| AKIEC | BCC | 25 | 0.410 |
| NV | BKL | 19 | 0.128 |
| BCC | SCCKA | 17 | 0.034 |
| MEL | BKL | 17 | 0.189 |
| NV | MEL | 17 | 0.114 |
| AKIEC | BKL | 16 | 0.262 |
| MEL | BCC | 10 | 0.111 |
| SCCKA | BKL | 10 | 0.105 |
| BCC | BKL | 8 | 0.016 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 17 | 0.034 |
| SCCKA | BKL | 10 | 0.105 |
| BCC | BKL | 8 | 0.016 |
| AKIEC | SCCKA | 7 | 0.115 |
| INF | BCC | 6 | 0.600 |
| BKL | SCCKA | 5 | 0.046 |
| SCCKA | AKIEC | 3 | 0.032 |
| BCC | AKIEC | 2 | 0.004 |
| INF | NV | 2 | 0.200 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
