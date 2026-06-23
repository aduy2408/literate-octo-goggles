# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_adaptive_gate_meta_gated_concat_weighted_sampler_focal
- backbone: tf_efficientnetv2_b2
- metadata_fusion: gated_concat
- image_fusion: adaptive_gate
- loss: focal
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

- accuracy: 0.7519083969465649
- balanced_accuracy: 0.5451728047437722
- dice_macro: 0.5343275139801965
- f1_macro: 0.5343275139801965
- roc_auc_macro_ovr: 0.9196855446460553
- top2_accuracy: 0.8874045801526718
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
| AKIEC | 61 | 0.446429 | 0.819672 | 0.937183 | 0.578035 | 0.956533 |
| BCC | 504 | 0.891892 | 0.85119 | 0.904412 | 0.871066 | 0.95833 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.850818 |
| BKL | 109 | 0.651163 | 0.513761 | 0.968051 | 0.574359 | 0.882024 |
| DF | 10 | 0.777778 | 0.7 | 0.998073 | 0.736842 | 0.990848 |
| INF | 10 | 0.25 | 0.2 | 0.99422 | 0.222222 | 0.956455 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.723231 |
| MEL | 90 | 0.712121 | 0.522222 | 0.980167 | 0.602564 | 0.932904 |
| NV | 149 | 0.757396 | 0.85906 | 0.954394 | 0.805031 | 0.968936 |
| SCCKA | 95 | 0.61 | 0.642105 | 0.959077 | 0.625641 | 0.928116 |
| VASC | 9 | 0.7 | 0.777778 | 0.997113 | 0.736842 | 0.968346 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.723231 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.850818 |
| INF | 10 | 0.25 | 0.2 | 0.99422 | 0.222222 | 0.956455 |
| BKL | 109 | 0.651163 | 0.513761 | 0.968051 | 0.574359 | 0.882024 |
| AKIEC | 61 | 0.446429 | 0.819672 | 0.937183 | 0.578035 | 0.956533 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 112 | 0.1113 |
| BCC | 481 | 0.3801 |
| BEN_OTH | 7 | 0.0165 |
| BKL | 86 | 0.1362 |
| DF | 9 | 0.0099 |
| INF | 8 | 0.0098 |
| MAL_OTH | 0 | 0.0009 |
| MEL | 66 | 0.0749 |
| NV | 169 | 0.1469 |
| SCCKA | 100 | 0.1026 |
| VASC | 10 | 0.0110 |

- mean_confidence: 0.7228118777275085
- median_confidence: 0.7537851929664612
- mean_top1_top2_gap: 0.5483698844909668
- mean_entropy: 0.7709335684776306
- low_confidence_rows: 153

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 30 | 0.333 |
| BCC | AKIEC | 28 | 0.056 |
| BCC | SCCKA | 23 | 0.046 |
| BKL | BCC | 23 | 0.211 |
| SCCKA | AKIEC | 18 | 0.189 |
| BCC | BKL | 11 | 0.022 |
| BKL | SCCKA | 11 | 0.101 |
| BKL | AKIEC | 10 | 0.092 |
| SCCKA | BCC | 10 | 0.105 |
| NV | MEL | 8 | 0.054 |
| BKL | NV | 6 | 0.055 |
| INF | BCC | 5 | 0.500 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 28 | 0.056 |
| BCC | SCCKA | 23 | 0.046 |
| SCCKA | AKIEC | 18 | 0.189 |
| BCC | BKL | 11 | 0.022 |
| BKL | SCCKA | 11 | 0.101 |
| INF | BCC | 5 | 0.500 |
| SCCKA | BKL | 5 | 0.053 |
| AKIEC | SCCKA | 3 | 0.049 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 62.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
