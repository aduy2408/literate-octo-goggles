# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/single_encoder_split_runs/singleenc_canvas_nometa_clean
- backbone: efficientnet_b2
- metadata_fusion: concat
- image_fusion: single_encoder_canvas
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

- accuracy: 0.7204198473282443
- balanced_accuracy: 0.4426356310134963
- dice_macro: 0.45719411465386484
- f1_macro: 0.45719411465386484
- roc_auc_macro_ovr: 0.8749101635994393
- top2_accuracy: 0.8587786259541985
- top3_accuracy: 0.9360687022900763

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
| AKIEC | 61 | 0.55102 | 0.442623 | 0.97771 | 0.490909 | 0.921604 | 0.827824 |
| BCC | 504 | 0.816254 | 0.916667 | 0.808824 | 0.863551 | 0.937347 | 0.930196 |
| BEN_OTH | 9 | 0.285714 | 0.222222 | 0.995188 | 0.25 | 0.91926 | 0.569618 |
| BKL | 109 | 0.505495 | 0.422018 | 0.952077 | 0.46 | 0.83878 | 0.794426 |
| DF | 10 | 0.6 | 0.3 | 0.998073 | 0.4 | 0.982659 | 0.796717 |
| INF | 10 | 0.166667 | 0.1 | 0.995183 | 0.125 | 0.924952 | 0.807152 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.336998 |  |
| MEL | 90 | 0.589041 | 0.477778 | 0.968685 | 0.527607 | 0.917386 | 0.808149 |
| NV | 149 | 0.712418 | 0.731544 | 0.951057 | 0.721854 | 0.950713 | 0.923268 |
| SCCKA | 95 | 0.651163 | 0.589474 | 0.96852 | 0.618785 | 0.927465 | 0.887878 |
| VASC | 9 | 0.5 | 0.666667 | 0.994225 | 0.571429 | 0.966848 | 0.823984 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.336998 |  |
| INF | 10 | 0.166667 | 0.1 | 0.995183 | 0.125 | 0.924952 | 0.807152 |
| BEN_OTH | 9 | 0.285714 | 0.222222 | 0.995188 | 0.25 | 0.91926 | 0.569618 |
| DF | 10 | 0.6 | 0.3 | 0.998073 | 0.4 | 0.982659 | 0.796717 |
| BKL | 109 | 0.505495 | 0.422018 | 0.952077 | 0.46 | 0.83878 | 0.794426 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 49 | 0.0544 |
| BCC | 566 | 0.5061 |
| BEN_OTH | 7 | 0.0111 |
| BKL | 91 | 0.0953 |
| DF | 5 | 0.0063 |
| INF | 6 | 0.0068 |
| MAL_OTH | 0 | 0.0007 |
| MEL | 73 | 0.0705 |
| NV | 153 | 0.1494 |
| SCCKA | 86 | 0.0887 |
| VASC | 12 | 0.0107 |

- mean_confidence: 0.8599659204483032
- median_confidence: 0.9545775055885315
- mean_top1_top2_gap: 0.7656934857368469
- mean_entropy: 0.4010350704193115
- low_confidence_rows: 66

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 34 | 0.312 |
| MEL | NV | 29 | 0.322 |
| SCCKA | BCC | 22 | 0.232 |
| NV | MEL | 17 | 0.114 |
| AKIEC | BCC | 12 | 0.197 |
| BCC | BKL | 12 | 0.024 |
| AKIEC | BKL | 10 | 0.164 |
| AKIEC | SCCKA | 10 | 0.164 |
| BKL | SCCKA | 10 | 0.092 |
| SCCKA | AKIEC | 10 | 0.105 |
| BCC | SCCKA | 9 | 0.018 |
| MEL | BCC | 9 | 0.100 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 12 | 0.024 |
| AKIEC | SCCKA | 10 | 0.164 |
| BKL | SCCKA | 10 | 0.092 |
| SCCKA | AKIEC | 10 | 0.105 |
| BCC | SCCKA | 9 | 0.018 |
| INF | BCC | 8 | 0.800 |
| BCC | AKIEC | 7 | 0.014 |
| SCCKA | BKL | 7 | 0.074 |
| INF | NV | 1 | 0.100 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
