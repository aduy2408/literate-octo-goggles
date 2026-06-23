# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_balanced_3xhead_ce_pretrained
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ce
- class_weight: False
- weighted_sampler: False
- balance_mode: hybrid
- balance_head_ratio: 3.0
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

- accuracy: 0.7662213740458015
- balanced_accuracy: 0.5762065550136731
- dice_macro: 0.5499193047228725
- f1_macro: 0.5499193047228725
- roc_auc_macro_ovr: 0.8967991796536808
- top2_accuracy: 0.8959923664122137
- top3_accuracy: 0.9408396946564885

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
| AKIEC | 61 | 0.566667 | 0.557377 | 0.973658 | 0.561983 | 0.951517 |
| BCC | 504 | 0.880392 | 0.890873 | 0.887868 | 0.885602 | 0.955875 |
| BEN_OTH | 9 | 0.285714 | 0.222222 | 0.995188 | 0.25 | 0.784194 |
| BKL | 109 | 0.716216 | 0.486239 | 0.977636 | 0.579235 | 0.88982 |
| DF | 10 | 0.642857 | 0.9 | 0.995183 | 0.75 | 0.994316 |
| INF | 10 | 0.192308 | 0.5 | 0.979769 | 0.277778 | 0.856166 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.586998 |
| MEL | 90 | 0.695652 | 0.533333 | 0.978079 | 0.603774 | 0.93592 |
| NV | 149 | 0.767857 | 0.865772 | 0.956618 | 0.81388 | 0.976559 |
| SCCKA | 95 | 0.612613 | 0.715789 | 0.954879 | 0.660194 | 0.944762 |
| VASC | 9 | 0.666667 | 0.666667 | 0.997113 | 0.666667 | 0.988664 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.586998 |
| BEN_OTH | 9 | 0.285714 | 0.222222 | 0.995188 | 0.25 | 0.784194 |
| INF | 10 | 0.192308 | 0.5 | 0.979769 | 0.277778 | 0.856166 |
| AKIEC | 61 | 0.566667 | 0.557377 | 0.973658 | 0.561983 | 0.951517 |
| BKL | 109 | 0.716216 | 0.486239 | 0.977636 | 0.579235 | 0.88982 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 60 | 0.0666 |
| BCC | 510 | 0.4525 |
| BEN_OTH | 7 | 0.0092 |
| BKL | 74 | 0.0844 |
| DF | 14 | 0.0157 |
| INF | 26 | 0.0265 |
| MAL_OTH | 0 | 0.0004 |
| MEL | 69 | 0.0714 |
| NV | 168 | 0.1588 |
| SCCKA | 111 | 0.1048 |
| VASC | 9 | 0.0096 |

- mean_confidence: 0.8712645173072815
- median_confidence: 0.96708744764328
- mean_top1_top2_gap: 0.7890195250511169
- mean_entropy: 0.3810822665691376
- low_confidence_rows: 69

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 27 | 0.300 |
| BKL | BCC | 24 | 0.220 |
| BCC | SCCKA | 16 | 0.032 |
| SCCKA | BCC | 15 | 0.158 |
| AKIEC | SCCKA | 13 | 0.213 |
| BCC | INF | 13 | 0.026 |
| BKL | SCCKA | 12 | 0.110 |
| BCC | AKIEC | 10 | 0.020 |
| NV | MEL | 9 | 0.060 |
| BKL | AKIEC | 7 | 0.064 |
| SCCKA | AKIEC | 7 | 0.074 |
| AKIEC | BKL | 6 | 0.098 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 16 | 0.032 |
| AKIEC | SCCKA | 13 | 0.213 |
| BKL | SCCKA | 12 | 0.110 |
| BCC | AKIEC | 10 | 0.020 |
| SCCKA | AKIEC | 7 | 0.074 |
| SCCKA | BKL | 4 | 0.042 |
| BCC | BKL | 3 | 0.006 |
| INF | BCC | 3 | 0.300 |

## Warnings

- [high] tail_precision_low: INF recall=0.500 but precision=0.192.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
