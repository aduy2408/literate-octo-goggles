# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine2_no_metadata_compact8/effv2b2_ldam_no_metadata_concat_simple
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

- accuracy: 0.7604961832061069
- balanced_accuracy: 0.4925701718339138
- dice_macro: 0.5103300057317682
- f1_macro: 0.5103300057317682
- roc_auc_macro_ovr: 0.904846575257769
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
| AKIEC | 61 | 0.532258 | 0.540984 | 0.970618 | 0.536585 | 0.956151 |
| BCC | 504 | 0.869565 | 0.912698 | 0.873162 | 0.89061 | 0.955554 |
| BEN_OTH | 9 | 0 | 0 | 1 | 0 | 0.808363 |
| BKL | 109 | 0.513043 | 0.541284 | 0.940362 | 0.526786 | 0.886459 |
| DF | 10 | 0.857143 | 0.6 | 0.999037 | 0.705882 | 0.973507 |
| INF | 10 | 1 | 0.1 | 1 | 0.181818 | 0.96869 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.59847 |
| MEL | 90 | 0.754386 | 0.477778 | 0.985386 | 0.585034 | 0.924484 |
| NV | 149 | 0.724324 | 0.899329 | 0.94327 | 0.802395 | 0.968765 |
| SCCKA | 95 | 0.650602 | 0.568421 | 0.96957 | 0.606742 | 0.929552 |
| VASC | 9 | 0.777778 | 0.777778 | 0.998075 | 0.777778 | 0.983317 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.59847 |
| BEN_OTH | 9 | 0 | 0 | 1 | 0 | 0.808363 |
| INF | 10 | 1 | 0.1 | 1 | 0.181818 | 0.96869 |
| BKL | 109 | 0.513043 | 0.541284 | 0.940362 | 0.526786 | 0.886459 |
| AKIEC | 61 | 0.532258 | 0.540984 | 0.970618 | 0.536585 | 0.956151 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 62 | 0.0715 |
| BCC | 529 | 0.4485 |
| BEN_OTH | 0 | 0.0047 |
| BKL | 115 | 0.1371 |
| DF | 7 | 0.0074 |
| INF | 1 | 0.0060 |
| MAL_OTH | 0 | 0.0007 |
| MEL | 57 | 0.0698 |
| NV | 185 | 0.1574 |
| SCCKA | 83 | 0.0879 |
| VASC | 9 | 0.0089 |

- mean_confidence: 0.7805922627449036
- median_confidence: 0.8511292934417725
- mean_top1_top2_gap: 0.6537861227989197
- mean_entropy: 0.6550244092941284
- low_confidence_rows: 140

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 35 | 0.389 |
| BKL | BCC | 23 | 0.211 |
| AKIEC | BKL | 15 | 0.246 |
| SCCKA | BKL | 15 | 0.158 |
| BCC | BKL | 13 | 0.026 |
| BCC | SCCKA | 13 | 0.026 |
| SCCKA | BCC | 13 | 0.137 |
| BKL | AKIEC | 10 | 0.092 |
| SCCKA | AKIEC | 10 | 0.105 |
| BKL | NV | 9 | 0.083 |
| AKIEC | BCC | 8 | 0.131 |
| BCC | AKIEC | 8 | 0.016 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| SCCKA | BKL | 15 | 0.158 |
| BCC | BKL | 13 | 0.026 |
| BCC | SCCKA | 13 | 0.026 |
| SCCKA | AKIEC | 10 | 0.105 |
| BCC | AKIEC | 8 | 0.016 |
| BKL | SCCKA | 7 | 0.064 |
| INF | BCC | 6 | 0.600 |
| AKIEC | SCCKA | 5 | 0.082 |

## Warnings

- [high] tail_predicted_zero: BEN_OTH has zero predicted rows.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
