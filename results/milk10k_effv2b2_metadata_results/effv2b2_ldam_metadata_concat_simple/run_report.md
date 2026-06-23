# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine1_metadata_compact8/effv2b2_ldam_metadata_concat_simple
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ldam
- class_weight: False
- weighted_sampler: False
- augmented_data_dir: None
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7175572519083969
- balanced_accuracy: 0.555753422318539
- dice_macro: 0.520803458078314
- f1_macro: 0.520803458078314
- roc_auc_macro_ovr: 0.9151327837313442
- top2_accuracy: 0.8673664122137404
- top3_accuracy: 0.9312977099236641

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
| AKIEC | 61 | 0.412844 | 0.737705 | 0.935157 | 0.529412 | 0.954357 |
| BCC | 504 | 0.919622 | 0.771825 | 0.9375 | 0.839266 | 0.958479 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.828788 |
| BKL | 109 | 0.522523 | 0.53211 | 0.943557 | 0.527273 | 0.871286 |
| DF | 10 | 0.666667 | 0.6 | 0.99711 | 0.631579 | 0.973699 |
| INF | 10 | 0.227273 | 0.5 | 0.983622 | 0.3125 | 0.969942 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.670172 |
| MEL | 90 | 0.693333 | 0.577778 | 0.975992 | 0.630303 | 0.941997 |
| NV | 149 | 0.734104 | 0.852349 | 0.948832 | 0.78882 | 0.974461 |
| SCCKA | 95 | 0.568807 | 0.652632 | 0.950682 | 0.607843 | 0.928304 |
| VASC | 9 | 0.7 | 0.777778 | 0.997113 | 0.736842 | 0.994974 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.670172 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.828788 |
| INF | 10 | 0.227273 | 0.5 | 0.983622 | 0.3125 | 0.969942 |
| BKL | 109 | 0.522523 | 0.53211 | 0.943557 | 0.527273 | 0.871286 |
| AKIEC | 61 | 0.412844 | 0.737705 | 0.935157 | 0.529412 | 0.954357 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 109 | 0.1164 |
| BCC | 423 | 0.3383 |
| BEN_OTH | 7 | 0.0143 |
| BKL | 111 | 0.1292 |
| DF | 9 | 0.0139 |
| INF | 22 | 0.0238 |
| MAL_OTH | 0 | 0.0014 |
| MEL | 75 | 0.0824 |
| NV | 173 | 0.1587 |
| SCCKA | 109 | 0.1103 |
| VASC | 10 | 0.0113 |

- mean_confidence: 0.7480098605155945
- median_confidence: 0.7956472635269165
- mean_top1_top2_gap: 0.6028713583946228
- mean_entropy: 0.7301409840583801
- low_confidence_rows: 174

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 31 | 0.062 |
| BCC | BKL | 28 | 0.056 |
| BCC | SCCKA | 27 | 0.054 |
| MEL | NV | 27 | 0.300 |
| SCCKA | AKIEC | 16 | 0.168 |
| BKL | BCC | 15 | 0.138 |
| BKL | AKIEC | 14 | 0.128 |
| BKL | SCCKA | 10 | 0.092 |
| AKIEC | SCCKA | 9 | 0.148 |
| BCC | INF | 9 | 0.018 |
| SCCKA | BCC | 9 | 0.095 |
| BCC | MEL | 8 | 0.016 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 31 | 0.062 |
| BCC | BKL | 28 | 0.056 |
| BCC | SCCKA | 27 | 0.054 |
| SCCKA | AKIEC | 16 | 0.168 |
| BKL | SCCKA | 10 | 0.092 |
| AKIEC | SCCKA | 9 | 0.148 |
| SCCKA | BKL | 7 | 0.074 |
| INF | BCC | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 86.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
