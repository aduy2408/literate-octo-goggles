# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_hybrid_tail_ablation/report_effv2b2_meta_concat_focal_hybrid_clean
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: focal
- class_weight: False
- weighted_sampler: False
- balance_mode: hybrid
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

- accuracy: 0.7595419847328244
- balanced_accuracy: 0.5387002827429274
- dice_macro: 0.5470466265680682
- f1_macro: 0.5470466265680682
- roc_auc_macro_ovr: 0.9199932903343484
- top2_accuracy: 0.8940839694656488
- top3_accuracy: 0.9437022900763359

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
| AKIEC | 61 | 0.449275 | 0.508197 | 0.961499 | 0.476923 | 0.945222 | 0.63101 |
| BCC | 504 | 0.858223 | 0.900794 | 0.862132 | 0.878993 | 0.953132 | 0.739395 |
| BEN_OTH | 9 | 0.222222 | 0.222222 | 0.993263 | 0.222222 | 0.801091 | 0.411976 |
| BKL | 109 | 0.650602 | 0.495413 | 0.969116 | 0.5625 | 0.881691 | 0.64136 |
| DF | 10 | 0.8 | 0.8 | 0.998073 | 0.8 | 0.945472 | 0.716677 |
| INF | 10 | 0.230769 | 0.3 | 0.990366 | 0.26087 | 0.958767 | 0.450417 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.82935 |  |
| MEL | 90 | 0.754098 | 0.511111 | 0.984342 | 0.609272 | 0.935003 | 0.766731 |
| NV | 149 | 0.770588 | 0.879195 | 0.956618 | 0.821317 | 0.972356 | 0.785134 |
| SCCKA | 95 | 0.628866 | 0.642105 | 0.962225 | 0.635417 | 0.934843 | 0.625475 |
| VASC | 9 | 0.857143 | 0.666667 | 0.999038 | 0.75 | 0.962999 | 0.839275 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.82935 |  |
| BEN_OTH | 9 | 0.222222 | 0.222222 | 0.993263 | 0.222222 | 0.801091 | 0.411976 |
| INF | 10 | 0.230769 | 0.3 | 0.990366 | 0.26087 | 0.958767 | 0.450417 |
| AKIEC | 61 | 0.449275 | 0.508197 | 0.961499 | 0.476923 | 0.945222 | 0.63101 |
| BKL | 109 | 0.650602 | 0.495413 | 0.969116 | 0.5625 | 0.881691 | 0.64136 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 69 | 0.0964 |
| BCC | 529 | 0.3959 |
| BEN_OTH | 9 | 0.0183 |
| BKL | 83 | 0.1301 |
| DF | 10 | 0.0160 |
| INF | 13 | 0.0220 |
| MAL_OTH | 0 | 0.0006 |
| MEL | 61 | 0.0756 |
| NV | 170 | 0.1449 |
| SCCKA | 97 | 0.0910 |
| VASC | 7 | 0.0091 |

- mean_confidence: 0.6811385750770569
- median_confidence: 0.7196207046508789
- mean_top1_top2_gap: 0.5056516528129578
- mean_entropy: 0.9260963797569275
- low_confidence_rows: 212

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 28 | 0.257 |
| MEL | NV | 28 | 0.311 |
| SCCKA | BCC | 15 | 0.158 |
| BCC | SCCKA | 14 | 0.028 |
| SCCKA | AKIEC | 14 | 0.147 |
| AKIEC | BCC | 13 | 0.213 |
| BCC | AKIEC | 12 | 0.024 |
| BKL | SCCKA | 12 | 0.110 |
| AKIEC | SCCKA | 10 | 0.164 |
| BKL | AKIEC | 9 | 0.083 |
| MEL | BCC | 7 | 0.078 |
| NV | BKL | 7 | 0.047 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 14 | 0.028 |
| SCCKA | AKIEC | 14 | 0.147 |
| BCC | AKIEC | 12 | 0.024 |
| BKL | SCCKA | 12 | 0.110 |
| AKIEC | SCCKA | 10 | 0.164 |
| BCC | BKL | 6 | 0.012 |
| INF | BCC | 5 | 0.500 |
| SCCKA | BKL | 5 | 0.053 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
