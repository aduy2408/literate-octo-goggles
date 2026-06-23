# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine1_metadata_compact8/effv2b2_focal_sampler05_metadata_concat_simple
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: focal
- class_weight: False
- weighted_sampler: True
- augmented_data_dir: None
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7404580152671756
- balanced_accuracy: 0.5518277776580842
- dice_macro: 0.5512127519617049
- f1_macro: 0.5512127519617049
- roc_auc_macro_ovr: 0.9191964732479853
- top2_accuracy: 0.892175572519084
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
| AKIEC | 61 | 0.455556 | 0.672131 | 0.950355 | 0.543046 | 0.954391 |
| BCC | 504 | 0.909091 | 0.833333 | 0.922794 | 0.869565 | 0.958519 |
| BEN_OTH | 9 | 0.333333 | 0.222222 | 0.99615 | 0.266667 | 0.874238 |
| BKL | 109 | 0.516949 | 0.559633 | 0.939297 | 0.537445 | 0.862288 |
| DF | 10 | 0.666667 | 0.8 | 0.996146 | 0.727273 | 0.988054 |
| INF | 10 | 0.4 | 0.4 | 0.99422 | 0.4 | 0.931214 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.710325 |
| MEL | 90 | 0.666667 | 0.577778 | 0.97286 | 0.619048 | 0.937416 |
| NV | 149 | 0.748503 | 0.838926 | 0.953281 | 0.791139 | 0.974909 |
| SCCKA | 95 | 0.58 | 0.610526 | 0.955929 | 0.594872 | 0.925797 |
| VASC | 9 | 1 | 0.555556 | 1 | 0.714286 | 0.994011 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.710325 |
| BEN_OTH | 9 | 0.333333 | 0.222222 | 0.99615 | 0.266667 | 0.874238 |
| INF | 10 | 0.4 | 0.4 | 0.99422 | 0.4 | 0.931214 |
| BKL | 109 | 0.516949 | 0.559633 | 0.939297 | 0.537445 | 0.862288 |
| AKIEC | 61 | 0.455556 | 0.672131 | 0.950355 | 0.543046 | 0.954391 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 90 | 0.1064 |
| BCC | 462 | 0.3425 |
| BEN_OTH | 6 | 0.0133 |
| BKL | 118 | 0.1684 |
| DF | 12 | 0.0144 |
| INF | 10 | 0.0123 |
| MAL_OTH | 0 | 0.0007 |
| MEL | 78 | 0.0881 |
| NV | 167 | 0.1408 |
| SCCKA | 100 | 0.1059 |
| VASC | 5 | 0.0071 |

- mean_confidence: 0.666560173034668
- median_confidence: 0.6767007112503052
- mean_top1_top2_gap: 0.45857056975364685
- mean_entropy: 0.8898169994354248
- low_confidence_rows: 205

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 29 | 0.058 |
| MEL | NV | 26 | 0.289 |
| BCC | SCCKA | 21 | 0.042 |
| BCC | AKIEC | 20 | 0.040 |
| SCCKA | AKIEC | 19 | 0.200 |
| BKL | BCC | 17 | 0.156 |
| NV | MEL | 15 | 0.101 |
| BKL | SCCKA | 12 | 0.110 |
| SCCKA | BKL | 10 | 0.105 |
| AKIEC | BKL | 9 | 0.148 |
| BKL | AKIEC | 9 | 0.083 |
| AKIEC | SCCKA | 7 | 0.115 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 29 | 0.058 |
| BCC | SCCKA | 21 | 0.042 |
| BCC | AKIEC | 20 | 0.040 |
| SCCKA | AKIEC | 19 | 0.200 |
| BKL | SCCKA | 12 | 0.110 |
| SCCKA | BKL | 10 | 0.105 |
| AKIEC | SCCKA | 7 | 0.115 |
| INF | BCC | 3 | 0.300 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 70.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
