# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine3_fusion_loss/report_effv2b2_meta_concat_ce_s05_clean
- backbone: tf_efficientnetv2_b2
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

- accuracy: 0.7547709923664122
- balanced_accuracy: 0.5666077611442839
- dice_macro: 0.5602218879821623
- f1_macro: 0.5602218879821623
- roc_auc_macro_ovr: 0.9363086679217603
- top2_accuracy: 0.8959923664122137
- top3_accuracy: 0.9465648854961832

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
| AKIEC | 61 | 0.411765 | 0.688525 | 0.93921 | 0.515337 | 0.94964 |
| BCC | 504 | 0.892784 | 0.859127 | 0.904412 | 0.875632 | 0.958691 |
| BEN_OTH | 9 | 0.25 | 0.111111 | 0.997113 | 0.153846 | 0.910598 |
| BKL | 109 | 0.597826 | 0.504587 | 0.960596 | 0.547264 | 0.865961 |
| DF | 10 | 0.8 | 0.8 | 0.998073 | 0.8 | 0.984297 |
| INF | 10 | 0.5 | 0.4 | 0.996146 | 0.444444 | 0.936127 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.881931 |
| MEL | 90 | 0.701299 | 0.6 | 0.975992 | 0.646707 | 0.945059 |
| NV | 149 | 0.806452 | 0.838926 | 0.96663 | 0.822368 | 0.975655 |
| SCCKA | 95 | 0.590476 | 0.652632 | 0.954879 | 0.62 | 0.925764 |
| VASC | 9 | 0.7 | 0.777778 | 0.997113 | 0.736842 | 0.965672 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.881931 |
| BEN_OTH | 9 | 0.25 | 0.111111 | 0.997113 | 0.153846 | 0.910598 |
| INF | 10 | 0.5 | 0.4 | 0.996146 | 0.444444 | 0.936127 |
| AKIEC | 61 | 0.411765 | 0.688525 | 0.93921 | 0.515337 | 0.94964 |
| BKL | 109 | 0.597826 | 0.504587 | 0.960596 | 0.547264 | 0.865961 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 102 | 0.1051 |
| BCC | 485 | 0.4105 |
| BEN_OTH | 4 | 0.0091 |
| BKL | 92 | 0.1107 |
| DF | 10 | 0.0115 |
| INF | 8 | 0.0077 |
| MAL_OTH | 0 | 0.0005 |
| MEL | 77 | 0.0812 |
| NV | 155 | 0.1425 |
| SCCKA | 105 | 0.1115 |
| VASC | 10 | 0.0098 |

- mean_confidence: 0.8042776584625244
- median_confidence: 0.8722440600395203
- mean_top1_top2_gap: 0.6771843433380127
- mean_entropy: 0.5498528480529785
- low_confidence_rows: 105

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 22 | 0.044 |
| MEL | NV | 22 | 0.244 |
| BCC | BKL | 18 | 0.036 |
| BKL | BCC | 18 | 0.165 |
| BCC | SCCKA | 17 | 0.034 |
| SCCKA | AKIEC | 16 | 0.168 |
| BKL | AKIEC | 15 | 0.138 |
| BKL | SCCKA | 15 | 0.138 |
| NV | MEL | 12 | 0.081 |
| SCCKA | BCC | 11 | 0.116 |
| AKIEC | SCCKA | 10 | 0.164 |
| AKIEC | BCC | 6 | 0.098 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 22 | 0.044 |
| BCC | BKL | 18 | 0.036 |
| BCC | SCCKA | 17 | 0.034 |
| SCCKA | AKIEC | 16 | 0.168 |
| BKL | SCCKA | 15 | 0.138 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | BKL | 6 | 0.063 |
| INF | BCC | 4 | 0.400 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 57.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
