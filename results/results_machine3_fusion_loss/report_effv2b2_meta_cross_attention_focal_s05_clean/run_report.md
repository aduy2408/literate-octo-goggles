# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine3_fusion_loss/report_effv2b2_meta_cross_attention_focal_s05_clean
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: cross_attention
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

- accuracy: 0.7614503816793893
- balanced_accuracy: 0.5569001577659721
- dice_macro: 0.5599427383342147
- f1_macro: 0.5599427383342147
- roc_auc_macro_ovr: 0.9343253599924212
- top2_accuracy: 0.9017175572519084
- top3_accuracy: 0.9484732824427481

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
| AKIEC | 61 | 0.473684 | 0.737705 | 0.949341 | 0.576923 | 0.953394 |
| BCC | 504 | 0.878049 | 0.857143 | 0.889706 | 0.86747 | 0.951651 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.86194 |
| BKL | 109 | 0.593407 | 0.495413 | 0.960596 | 0.54 | 0.877285 |
| DF | 10 | 0.8 | 0.8 | 0.998073 | 0.8 | 0.977842 |
| INF | 10 | 0.5 | 0.3 | 0.99711 | 0.375 | 0.943256 |
| MAL_OTH | 2 | 0 | 0 | 0.998088 | 0 | 0.897228 |
| MEL | 90 | 0.736842 | 0.622222 | 0.979123 | 0.674699 | 0.941069 |
| NV | 149 | 0.828025 | 0.872483 | 0.969967 | 0.849673 | 0.973968 |
| SCCKA | 95 | 0.59434 | 0.663158 | 0.954879 | 0.626866 | 0.931386 |
| VASC | 9 | 0.75 | 0.666667 | 0.998075 | 0.705882 | 0.96856 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 0.998088 | 0 | 0.897228 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.86194 |
| INF | 10 | 0.5 | 0.3 | 0.99711 | 0.375 | 0.943256 |
| BKL | 109 | 0.593407 | 0.495413 | 0.960596 | 0.54 | 0.877285 |
| AKIEC | 61 | 0.473684 | 0.737705 | 0.949341 | 0.576923 | 0.953394 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 95 | 0.1107 |
| BCC | 492 | 0.3694 |
| BEN_OTH | 5 | 0.0105 |
| BKL | 91 | 0.1468 |
| DF | 10 | 0.0141 |
| INF | 6 | 0.0064 |
| MAL_OTH | 2 | 0.0015 |
| MEL | 76 | 0.0921 |
| NV | 157 | 0.1330 |
| SCCKA | 106 | 0.1073 |
| VASC | 8 | 0.0082 |

- mean_confidence: 0.6919740438461304
- median_confidence: 0.706525444984436
- mean_top1_top2_gap: 0.49348413944244385
- mean_entropy: 0.8175482153892517
- low_confidence_rows: 177

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 24 | 0.220 |
| BCC | AKIEC | 23 | 0.046 |
| BCC | SCCKA | 20 | 0.040 |
| MEL | NV | 18 | 0.200 |
| BKL | SCCKA | 16 | 0.147 |
| BCC | BKL | 14 | 0.028 |
| SCCKA | AKIEC | 14 | 0.147 |
| SCCKA | BCC | 11 | 0.116 |
| BKL | AKIEC | 10 | 0.092 |
| NV | BKL | 8 | 0.054 |
| NV | MEL | 7 | 0.047 |
| AKIEC | BCC | 6 | 0.098 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 23 | 0.046 |
| BCC | SCCKA | 20 | 0.040 |
| BKL | SCCKA | 16 | 0.147 |
| BCC | BKL | 14 | 0.028 |
| SCCKA | AKIEC | 14 | 0.147 |
| AKIEC | SCCKA | 6 | 0.098 |
| SCCKA | BKL | 6 | 0.063 |
| INF | BCC | 5 | 0.500 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 57.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
