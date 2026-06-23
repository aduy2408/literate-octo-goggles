# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine3_fusion_loss/report_effv2b2_meta_co_attention_focal_s05_clean
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: co_attention
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

- accuracy: 0.7662213740458015
- balanced_accuracy: 0.5590009882201419
- dice_macro: 0.5535109249994838
- f1_macro: 0.5535109249994838
- roc_auc_macro_ovr: 0.936668487263157
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
| AKIEC | 61 | 0.514706 | 0.57377 | 0.966565 | 0.542636 | 0.94783 |
| BCC | 504 | 0.899384 | 0.869048 | 0.909926 | 0.883956 | 0.954471 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.886001 |
| BKL | 109 | 0.614583 | 0.541284 | 0.960596 | 0.57561 | 0.877099 |
| DF | 10 | 0.615385 | 0.8 | 0.995183 | 0.695652 | 0.980925 |
| INF | 10 | 0.625 | 0.5 | 0.99711 | 0.555556 | 0.946628 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.896272 |
| MEL | 90 | 0.617021 | 0.644444 | 0.962422 | 0.630435 | 0.935247 |
| NV | 149 | 0.825806 | 0.85906 | 0.969967 | 0.842105 | 0.97455 |
| SCCKA | 95 | 0.578947 | 0.694737 | 0.949633 | 0.631579 | 0.938444 |
| VASC | 9 | 0.625 | 0.555556 | 0.997113 | 0.588235 | 0.965886 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.896272 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.886001 |
| AKIEC | 61 | 0.514706 | 0.57377 | 0.966565 | 0.542636 | 0.94783 |
| INF | 10 | 0.625 | 0.5 | 0.99711 | 0.555556 | 0.946628 |
| BKL | 109 | 0.614583 | 0.541284 | 0.960596 | 0.57561 | 0.877099 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 68 | 0.0894 |
| BCC | 487 | 0.3807 |
| BEN_OTH | 5 | 0.0104 |
| BKL | 96 | 0.1334 |
| DF | 13 | 0.0159 |
| INF | 8 | 0.0110 |
| MAL_OTH | 0 | 0.0006 |
| MEL | 94 | 0.1064 |
| NV | 155 | 0.1354 |
| SCCKA | 114 | 0.1079 |
| VASC | 8 | 0.0090 |

- mean_confidence: 0.7219374775886536
- median_confidence: 0.7540470361709595
- mean_top1_top2_gap: 0.5392646789550781
- mean_entropy: 0.7483533620834351
- low_confidence_rows: 148

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 19 | 0.211 |
| BKL | BCC | 18 | 0.165 |
| BCC | AKIEC | 17 | 0.034 |
| BCC | SCCKA | 17 | 0.034 |
| BKL | SCCKA | 16 | 0.147 |
| AKIEC | SCCKA | 13 | 0.213 |
| BCC | BKL | 12 | 0.024 |
| NV | MEL | 12 | 0.081 |
| SCCKA | BKL | 10 | 0.105 |
| SCCKA | AKIEC | 9 | 0.095 |
| SCCKA | BCC | 9 | 0.095 |
| BCC | MEL | 8 | 0.016 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 17 | 0.034 |
| BCC | SCCKA | 17 | 0.034 |
| BKL | SCCKA | 16 | 0.147 |
| AKIEC | SCCKA | 13 | 0.213 |
| BCC | BKL | 12 | 0.024 |
| SCCKA | BKL | 10 | 0.105 |
| SCCKA | AKIEC | 9 | 0.095 |
| INF | BCC | 3 | 0.300 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 46.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
