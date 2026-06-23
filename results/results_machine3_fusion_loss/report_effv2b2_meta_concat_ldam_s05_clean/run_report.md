# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine3_fusion_loss/report_effv2b2_meta_concat_ldam_s05_clean
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ldam
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

- accuracy: 0.7395038167938931
- balanced_accuracy: 0.5569700149857879
- dice_macro: 0.5368041718044444
- f1_macro: 0.5368041718044444
- roc_auc_macro_ovr: 0.924277583976803
- top2_accuracy: 0.8721374045801527
- top3_accuracy: 0.9284351145038168

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
| AKIEC | 61 | 0.348214 | 0.639344 | 0.926039 | 0.450867 | 0.938811 |
| BCC | 504 | 0.898305 | 0.84127 | 0.911765 | 0.868852 | 0.949 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.871992 |
| BKL | 109 | 0.636364 | 0.449541 | 0.970181 | 0.526882 | 0.86346 |
| DF | 10 | 0.888889 | 0.8 | 0.999037 | 0.842105 | 0.963391 |
| INF | 10 | 0.4 | 0.4 | 0.99422 | 0.4 | 0.927842 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.830784 |
| MEL | 90 | 0.666667 | 0.577778 | 0.97286 | 0.619048 | 0.936952 |
| NV | 149 | 0.802548 | 0.845638 | 0.965517 | 0.823529 | 0.972677 |
| SCCKA | 95 | 0.570175 | 0.684211 | 0.948583 | 0.62201 | 0.928613 |
| VASC | 9 | 0.5 | 0.777778 | 0.993263 | 0.608696 | 0.983531 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.830784 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.871992 |
| INF | 10 | 0.4 | 0.4 | 0.99422 | 0.4 | 0.927842 |
| AKIEC | 61 | 0.348214 | 0.639344 | 0.926039 | 0.450867 | 0.938811 |
| BKL | 109 | 0.636364 | 0.449541 | 0.970181 | 0.526882 | 0.86346 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 112 | 0.1155 |
| BCC | 472 | 0.3822 |
| BEN_OTH | 5 | 0.0136 |
| BKL | 77 | 0.1123 |
| DF | 9 | 0.0152 |
| INF | 10 | 0.0132 |
| MAL_OTH | 0 | 0.0025 |
| MEL | 78 | 0.0811 |
| NV | 157 | 0.1406 |
| SCCKA | 114 | 0.1114 |
| VASC | 14 | 0.0124 |

- mean_confidence: 0.7510007619857788
- median_confidence: 0.8041555881500244
- mean_top1_top2_gap: 0.6111563444137573
- mean_entropy: 0.7209274768829346
- low_confidence_rows: 157

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 36 | 0.071 |
| MEL | NV | 22 | 0.244 |
| BKL | BCC | 21 | 0.193 |
| BCC | SCCKA | 17 | 0.034 |
| BKL | AKIEC | 17 | 0.156 |
| SCCKA | AKIEC | 16 | 0.168 |
| BKL | SCCKA | 15 | 0.138 |
| AKIEC | SCCKA | 13 | 0.213 |
| BCC | BKL | 10 | 0.020 |
| NV | MEL | 10 | 0.067 |
| SCCKA | BCC | 9 | 0.095 |
| BCC | MEL | 6 | 0.012 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 36 | 0.071 |
| BCC | SCCKA | 17 | 0.034 |
| SCCKA | AKIEC | 16 | 0.168 |
| BKL | SCCKA | 15 | 0.138 |
| AKIEC | SCCKA | 13 | 0.213 |
| BCC | BKL | 10 | 0.020 |
| INF | BCC | 4 | 0.400 |
| SCCKA | BKL | 4 | 0.042 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 63.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
