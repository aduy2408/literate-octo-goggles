# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine2_meta_aug_derm/report_effv2b2_meta_concat_focal_s05_aug_inf_benoth15_safe
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
- augmented_classes: ['INF', 'BEN_OTH']
- augmented_max_per_class: 15
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7681297709923665
- balanced_accuracy: 0.5375501724078189
- dice_macro: 0.5403382311194672
- f1_macro: 0.5403382311194672
- roc_auc_macro_ovr: 0.9252664859681258
- top2_accuracy: 0.8845419847328244
- top3_accuracy: 0.9341603053435115

## Data Distribution

### Train

- rows: 4222
- real_rows: 4192
- synthetic_rows: 30
- ignore_metadata_rows: 0

| class | count | synthetic |
|---|---:|---:|
| AKIEC | 242 | 0 |
| BCC | 2018 | 0 |
| BEN_OTH | 50 | 15 |
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
| AKIEC | 61 | 0.486486 | 0.590164 | 0.961499 | 0.533333 | 0.941701 |
| BCC | 504 | 0.883629 | 0.888889 | 0.891544 | 0.886251 | 0.951465 |
| BEN_OTH | 9 | 0.5 | 0.111111 | 0.999038 | 0.181818 | 0.91402 |
| BKL | 109 | 0.720588 | 0.449541 | 0.979766 | 0.553672 | 0.849508 |
| DF | 10 | 0.727273 | 0.8 | 0.99711 | 0.761905 | 0.942293 |
| INF | 10 | 0.272727 | 0.3 | 0.992293 | 0.285714 | 0.958767 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.841778 |
| MEL | 90 | 0.678571 | 0.633333 | 0.971816 | 0.655172 | 0.929865 |
| NV | 149 | 0.789157 | 0.879195 | 0.961068 | 0.831746 | 0.975551 |
| SCCKA | 95 | 0.567797 | 0.705263 | 0.946485 | 0.629108 | 0.930624 |
| VASC | 9 | 0.714286 | 0.555556 | 0.998075 | 0.625 | 0.942359 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.841778 |
| BEN_OTH | 9 | 0.5 | 0.111111 | 0.999038 | 0.181818 | 0.91402 |
| INF | 10 | 0.272727 | 0.3 | 0.992293 | 0.285714 | 0.958767 |
| AKIEC | 61 | 0.486486 | 0.590164 | 0.961499 | 0.533333 | 0.941701 |
| BKL | 109 | 0.720588 | 0.449541 | 0.979766 | 0.553672 | 0.849508 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 74 | 0.0999 |
| BCC | 507 | 0.3866 |
| BEN_OTH | 2 | 0.0098 |
| BKL | 68 | 0.1074 |
| DF | 11 | 0.0169 |
| INF | 11 | 0.0161 |
| MAL_OTH | 0 | 0.0024 |
| MEL | 84 | 0.0901 |
| NV | 166 | 0.1513 |
| SCCKA | 118 | 0.1107 |
| VASC | 7 | 0.0087 |

- mean_confidence: 0.6870476603507996
- median_confidence: 0.7180163860321045
- mean_top1_top2_gap: 0.512593150138855
- mean_entropy: 0.8928333520889282
- low_confidence_rows: 197

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 21 | 0.233 |
| BKL | BCC | 19 | 0.174 |
| BKL | SCCKA | 18 | 0.165 |
| BCC | SCCKA | 17 | 0.034 |
| BCC | AKIEC | 16 | 0.032 |
| AKIEC | SCCKA | 12 | 0.197 |
| SCCKA | BCC | 12 | 0.126 |
| SCCKA | AKIEC | 11 | 0.116 |
| AKIEC | BCC | 8 | 0.131 |
| BCC | BKL | 8 | 0.016 |
| BCC | MEL | 8 | 0.016 |
| BKL | MEL | 8 | 0.073 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | SCCKA | 18 | 0.165 |
| BCC | SCCKA | 17 | 0.034 |
| BCC | AKIEC | 16 | 0.032 |
| AKIEC | SCCKA | 12 | 0.197 |
| SCCKA | AKIEC | 11 | 0.116 |
| BCC | BKL | 8 | 0.016 |
| INF | BCC | 5 | 0.500 |
| SCCKA | BKL | 4 | 0.042 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 41.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
