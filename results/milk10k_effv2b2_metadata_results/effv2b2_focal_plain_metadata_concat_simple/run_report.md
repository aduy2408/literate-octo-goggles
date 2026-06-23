# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine1_metadata_compact8/effv2b2_focal_plain_metadata_concat_simple
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: focal
- class_weight: False
- weighted_sampler: False
- augmented_data_dir: None
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7490458015267175
- balanced_accuracy: 0.5183311736403384
- dice_macro: 0.5317107631151065
- f1_macro: 0.5317107631151065
- roc_auc_macro_ovr: 0.9216615223167955
- top2_accuracy: 0.9017175572519084
- top3_accuracy: 0.9513358778625954

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
| AKIEC | 61 | 0.523077 | 0.557377 | 0.968592 | 0.539683 | 0.953394 |
| BCC | 504 | 0.869732 | 0.900794 | 0.875 | 0.88499 | 0.958414 |
| BEN_OTH | 9 | 0.5 | 0.111111 | 0.999038 | 0.181818 | 0.833494 |
| BKL | 109 | 0.583333 | 0.513761 | 0.957401 | 0.546341 | 0.876132 |
| DF | 10 | 0.7 | 0.7 | 0.99711 | 0.7 | 0.973988 |
| INF | 10 | 0.363636 | 0.4 | 0.993256 | 0.380952 | 0.977649 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.735182 |
| MEL | 90 | 0.677966 | 0.444444 | 0.980167 | 0.536913 | 0.932139 |
| NV | 149 | 0.683417 | 0.912752 | 0.929922 | 0.781609 | 0.975163 |
| SCCKA | 95 | 0.61039 | 0.494737 | 0.96852 | 0.546512 | 0.927321 |
| VASC | 9 | 0.857143 | 0.666667 | 0.999038 | 0.75 | 0.995402 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.735182 |
| BEN_OTH | 9 | 0.5 | 0.111111 | 0.999038 | 0.181818 | 0.833494 |
| INF | 10 | 0.363636 | 0.4 | 0.993256 | 0.380952 | 0.977649 |
| MEL | 90 | 0.677966 | 0.444444 | 0.980167 | 0.536913 | 0.932139 |
| AKIEC | 61 | 0.523077 | 0.557377 | 0.968592 | 0.539683 | 0.953394 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 65 | 0.0778 |
| BCC | 522 | 0.4053 |
| BEN_OTH | 2 | 0.0084 |
| BKL | 96 | 0.1300 |
| DF | 10 | 0.0129 |
| INF | 11 | 0.0134 |
| MAL_OTH | 0 | 0.0018 |
| MEL | 59 | 0.0698 |
| NV | 199 | 0.1834 |
| SCCKA | 77 | 0.0890 |
| VASC | 7 | 0.0083 |

- mean_confidence: 0.720269501209259
- median_confidence: 0.7581461668014526
- mean_top1_top2_gap: 0.5534039735794067
- mean_entropy: 0.811059296131134
- low_confidence_rows: 147

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 40 | 0.444 |
| BKL | BCC | 24 | 0.220 |
| SCCKA | BCC | 20 | 0.211 |
| BCC | SCCKA | 15 | 0.030 |
| SCCKA | AKIEC | 14 | 0.147 |
| AKIEC | BKL | 13 | 0.213 |
| BKL | NV | 12 | 0.110 |
| SCCKA | BKL | 12 | 0.126 |
| BCC | AKIEC | 9 | 0.018 |
| AKIEC | SCCKA | 7 | 0.115 |
| BCC | BKL | 7 | 0.014 |
| BKL | AKIEC | 7 | 0.064 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 15 | 0.030 |
| SCCKA | AKIEC | 14 | 0.147 |
| SCCKA | BKL | 12 | 0.126 |
| BCC | AKIEC | 9 | 0.018 |
| AKIEC | SCCKA | 7 | 0.115 |
| BCC | BKL | 7 | 0.014 |
| BKL | SCCKA | 7 | 0.064 |
| INF | BCC | 3 | 0.300 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
