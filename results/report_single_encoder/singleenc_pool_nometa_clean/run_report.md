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

- accuracy: 0.6736641221374046
- balanced_accuracy: 0.3794547908899595
- dice_macro: 0.3958818744094628
- f1_macro: 0.3958818744094628
- roc_auc_macro_ovr: 0.8654417583052901
- top2_accuracy: 0.8234732824427481
- top3_accuracy: 0.9112595419847328

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
| AKIEC | 61 | 0.5 | 0.262295 | 0.983789 | 0.344086 | 0.910808 | 0.684883 |
| BCC | 504 | 0.786096 | 0.875 | 0.779412 | 0.828169 | 0.912669 | 0.880699 |
| BEN_OTH | 9 | 1 | 0.111111 | 1 | 0.2 | 0.857876 | 0.808157 |
| BKL | 109 | 0.415584 | 0.293578 | 0.952077 | 0.344086 | 0.788893 | 0.70871 |
| DF | 10 | 0.111111 | 0.1 | 0.992293 | 0.105263 | 0.855588 | 0.462921 |
| INF | 10 | 0.25 | 0.1 | 0.99711 | 0.142857 | 0.862524 | 0.321197 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.696463 |  |
| MEL | 90 | 0.513514 | 0.422222 | 0.962422 | 0.463415 | 0.861355 | 0.767756 |
| NV | 149 | 0.615789 | 0.785235 | 0.918799 | 0.690265 | 0.94489 | 0.880288 |
| SCCKA | 95 | 0.582418 | 0.557895 | 0.960126 | 0.569892 | 0.919158 | 0.829929 |
| VASC | 9 | 0.666667 | 0.666667 | 0.997113 | 0.666667 | 0.909635 | 0.868661 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.696463 |  |
| DF | 10 | 0.111111 | 0.1 | 0.992293 | 0.105263 | 0.855588 | 0.462921 |
| INF | 10 | 0.25 | 0.1 | 0.99711 | 0.142857 | 0.862524 | 0.321197 |
| BEN_OTH | 9 | 1 | 0.111111 | 1 | 0.2 | 0.857876 | 0.808157 |
| AKIEC | 61 | 0.5 | 0.262295 | 0.983789 | 0.344086 | 0.910808 | 0.684883 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 32 | 0.0417 |
| BCC | 561 | 0.4866 |
| BEN_OTH | 1 | 0.0077 |
| BKL | 77 | 0.0911 |
| DF | 9 | 0.0122 |
| INF | 4 | 0.0098 |
| MAL_OTH | 0 | 0.0006 |
| MEL | 74 | 0.0768 |
| NV | 190 | 0.1770 |
| SCCKA | 91 | 0.0871 |
| VASC | 9 | 0.0094 |

- mean_confidence: 0.7948505878448486
- median_confidence: 0.8715876936912537
- mean_top1_top2_gap: 0.6747757196426392
- mean_entropy: 0.6077224016189575
- low_confidence_rows: 129

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 38 | 0.349 |
| MEL | NV | 36 | 0.400 |
| SCCKA | BCC | 30 | 0.316 |
| AKIEC | BCC | 23 | 0.377 |
| BCC | SCCKA | 21 | 0.042 |
| BCC | NV | 16 | 0.032 |
| NV | MEL | 16 | 0.107 |
| AKIEC | BKL | 13 | 0.213 |
| BCC | BKL | 12 | 0.024 |
| BKL | MEL | 12 | 0.110 |
| BKL | NV | 12 | 0.110 |
| MEL | BCC | 9 | 0.100 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 21 | 0.042 |
| BCC | BKL | 12 | 0.024 |
| BKL | SCCKA | 8 | 0.073 |
| SCCKA | BKL | 8 | 0.084 |
| AKIEC | SCCKA | 7 | 0.115 |
| BCC | AKIEC | 5 | 0.010 |
| INF | BCC | 5 | 0.500 |
| SCCKA | AKIEC | 3 | 0.032 |
| INF | NV | 1 | 0.100 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
