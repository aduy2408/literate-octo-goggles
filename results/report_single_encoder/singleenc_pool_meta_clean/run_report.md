# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/single_encoder_split_runs/singleenc_pool_meta_clean
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

- accuracy: 0.7108778625954199
- balanced_accuracy: 0.4243347533948689
- dice_macro: 0.45318287891961967
- f1_macro: 0.45318287891961967
- roc_auc_macro_ovr: 0.9174702018794881
- top2_accuracy: 0.8692748091603053
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

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AKIEC | 61 | 0.590909 | 0.213115 | 0.990881 | 0.313253 | 0.942515 | 0.597855 |
| BCC | 504 | 0.798951 | 0.906746 | 0.788603 | 0.849442 | 0.925063 | 0.865497 |
| BEN_OTH | 9 | 1 | 0.111111 | 1 | 0.2 | 0.877232 | 0.482916 |
| BKL | 109 | 0.445652 | 0.376147 | 0.945687 | 0.40796 | 0.78903 | 0.698416 |
| DF | 10 | 0.75 | 0.3 | 0.999037 | 0.428571 | 0.985645 | 0.560032 |
| INF | 10 | 0.333333 | 0.1 | 0.998073 | 0.153846 | 0.953565 | 0.335535 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.844168 |  |
| MEL | 90 | 0.619048 | 0.433333 | 0.974948 | 0.509804 | 0.901276 | 0.72337 |
| NV | 149 | 0.730994 | 0.838926 | 0.948832 | 0.78125 | 0.963531 | 0.822151 |
| SCCKA | 95 | 0.522523 | 0.610526 | 0.944386 | 0.563107 | 0.930358 | 0.817618 |
| VASC | 9 | 0.777778 | 0.777778 | 0.998075 | 0.777778 | 0.979788 | 0.627882 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.844168 |  |
| INF | 10 | 0.333333 | 0.1 | 0.998073 | 0.153846 | 0.953565 | 0.335535 |
| BEN_OTH | 9 | 1 | 0.111111 | 1 | 0.2 | 0.877232 | 0.482916 |
| AKIEC | 61 | 0.590909 | 0.213115 | 0.990881 | 0.313253 | 0.942515 | 0.597855 |
| BKL | 109 | 0.445652 | 0.376147 | 0.945687 | 0.40796 | 0.78903 | 0.698416 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 22 | 0.0426 |
| BCC | 572 | 0.4831 |
| BEN_OTH | 1 | 0.0051 |
| BKL | 92 | 0.1090 |
| DF | 4 | 0.0063 |
| INF | 3 | 0.0095 |
| MAL_OTH | 0 | 0.0015 |
| MEL | 63 | 0.0723 |
| NV | 171 | 0.1524 |
| SCCKA | 111 | 0.1081 |
| VASC | 9 | 0.0101 |

- mean_confidence: 0.7764731645584106
- median_confidence: 0.8540850877761841
- mean_top1_top2_gap: 0.6493982076644897
- mean_entropy: 0.6740277409553528
- low_confidence_rows: 152

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 39 | 0.358 |
| MEL | NV | 30 | 0.333 |
| BCC | SCCKA | 28 | 0.056 |
| SCCKA | BCC | 26 | 0.274 |
| AKIEC | BCC | 22 | 0.361 |
| AKIEC | BKL | 14 | 0.230 |
| AKIEC | SCCKA | 12 | 0.197 |
| BKL | SCCKA | 11 | 0.101 |
| MEL | BKL | 10 | 0.111 |
| NV | BKL | 10 | 0.067 |
| MEL | BCC | 9 | 0.100 |
| BKL | MEL | 8 | 0.073 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 28 | 0.056 |
| AKIEC | SCCKA | 12 | 0.197 |
| BKL | SCCKA | 11 | 0.101 |
| SCCKA | BKL | 8 | 0.084 |
| BCC | BKL | 7 | 0.014 |
| INF | BCC | 6 | 0.600 |
| BCC | AKIEC | 3 | 0.006 |
| SCCKA | AKIEC | 3 | 0.032 |
| INF | NV | 1 | 0.100 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
