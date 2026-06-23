# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine1_backbones/report_backbone_convnext_base_nometa_concat_focal_s05_clean
- backbone: convnext_base
- metadata_fusion: concat
- image_fusion: concat
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

- accuracy: 0.773854961832061
- balanced_accuracy: 0.5493026642962063
- dice_macro: 0.5518263665786365
- f1_macro: 0.5518263665786365
- roc_auc_macro_ovr: 0.937392654586159
- top2_accuracy: 0.9112595419847328
- top3_accuracy: 0.9570610687022901

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
| AKIEC | 61 | 0.57377 | 0.57377 | 0.973658 | 0.57377 | 0.95851 |
| BCC | 504 | 0.886497 | 0.89881 | 0.893382 | 0.892611 | 0.961915 |
| BEN_OTH | 9 | 0 | 0 | 0.999038 | 0 | 0.898193 |
| BKL | 109 | 0.584906 | 0.568807 | 0.953142 | 0.576744 | 0.896132 |
| DF | 10 | 0.692308 | 0.9 | 0.996146 | 0.782609 | 0.99711 |
| INF | 10 | 1 | 0.2 | 1 | 0.333333 | 0.957418 |
| MAL_OTH | 2 | 0 | 0 | 0.999044 | 0 | 0.800191 |
| MEL | 90 | 0.655556 | 0.655556 | 0.967641 | 0.655556 | 0.937404 |
| NV | 149 | 0.842466 | 0.825503 | 0.974416 | 0.833898 | 0.977813 |
| SCCKA | 95 | 0.559633 | 0.642105 | 0.949633 | 0.598039 | 0.944706 |
| VASC | 9 | 0.875 | 0.777778 | 0.999038 | 0.823529 | 0.981927 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 0.999044 | 0 | 0.800191 |
| BEN_OTH | 9 | 0 | 0 | 0.999038 | 0 | 0.898193 |
| INF | 10 | 1 | 0.2 | 1 | 0.333333 | 0.957418 |
| AKIEC | 61 | 0.57377 | 0.57377 | 0.973658 | 0.57377 | 0.95851 |
| BKL | 109 | 0.584906 | 0.568807 | 0.953142 | 0.576744 | 0.896132 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 61 | 0.0591 |
| BCC | 511 | 0.4490 |
| BEN_OTH | 1 | 0.0054 |
| BKL | 106 | 0.1213 |
| DF | 13 | 0.0122 |
| INF | 2 | 0.0047 |
| MAL_OTH | 1 | 0.0011 |
| MEL | 90 | 0.0960 |
| NV | 146 | 0.1391 |
| SCCKA | 109 | 0.1053 |
| VASC | 8 | 0.0068 |

- mean_confidence: 0.8422402143478394
- median_confidence: 0.9031976461410522
- mean_top1_top2_gap: 0.7268039584159851
- mean_entropy: 0.45363593101501465
- low_confidence_rows: 54

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 19 | 0.174 |
| BKL | SCCKA | 17 | 0.156 |
| MEL | NV | 17 | 0.189 |
| BCC | SCCKA | 16 | 0.032 |
| NV | MEL | 16 | 0.107 |
| BCC | BKL | 13 | 0.026 |
| SCCKA | BCC | 13 | 0.137 |
| AKIEC | SCCKA | 12 | 0.197 |
| BCC | AKIEC | 12 | 0.024 |
| SCCKA | BKL | 12 | 0.126 |
| AKIEC | BKL | 8 | 0.131 |
| SCCKA | AKIEC | 7 | 0.074 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | SCCKA | 17 | 0.156 |
| BCC | SCCKA | 16 | 0.032 |
| BCC | BKL | 13 | 0.026 |
| AKIEC | SCCKA | 12 | 0.197 |
| BCC | AKIEC | 12 | 0.024 |
| SCCKA | BKL | 12 | 0.126 |
| SCCKA | AKIEC | 7 | 0.074 |
| INF | BCC | 6 | 0.600 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 41.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
