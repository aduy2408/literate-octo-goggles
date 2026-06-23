# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_balanced_focal
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

- accuracy: 0.7061068702290076
- balanced_accuracy: 0.4409260571098569
- dice_macro: 0.4592993006664745
- f1_macro: 0.4592993006664745
- roc_auc_macro_ovr: 0.8996586581967726
- top2_accuracy: 0.8597328244274809
- top3_accuracy: 0.9398854961832062

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
| AKIEC | 61 | 0.438356 | 0.52459 | 0.95846 | 0.477612 | 0.932782 |
| BCC | 504 | 0.881607 | 0.827381 | 0.897059 | 0.853634 | 0.948588 |
| BEN_OTH | 9 | 0.333333 | 0.111111 | 0.998075 | 0.166667 | 0.869212 |
| BKL | 109 | 0.464 | 0.53211 | 0.928647 | 0.495726 | 0.854569 |
| DF | 10 | 0.5 | 0.1 | 0.999037 | 0.166667 | 0.930829 |
| INF | 10 | 1 | 0.2 | 1 | 0.333333 | 0.905299 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.745698 |
| MEL | 90 | 0.606557 | 0.411111 | 0.974948 | 0.490066 | 0.914393 |
| NV | 149 | 0.707865 | 0.845638 | 0.942158 | 0.770642 | 0.964674 |
| SCCKA | 95 | 0.483871 | 0.631579 | 0.932844 | 0.547945 | 0.907307 |
| VASC | 9 | 0.857143 | 0.666667 | 0.999038 | 0.75 | 0.922896 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.745698 |
| BEN_OTH | 9 | 0.333333 | 0.111111 | 0.998075 | 0.166667 | 0.869212 |
| DF | 10 | 0.5 | 0.1 | 0.999037 | 0.166667 | 0.930829 |
| INF | 10 | 1 | 0.2 | 1 | 0.333333 | 0.905299 |
| AKIEC | 61 | 0.438356 | 0.52459 | 0.95846 | 0.477612 | 0.932782 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 73 | 0.0801 |
| BCC | 473 | 0.4074 |
| BEN_OTH | 3 | 0.0035 |
| BKL | 125 | 0.1460 |
| DF | 2 | 0.0022 |
| INF | 2 | 0.0019 |
| MAL_OTH | 0 | 0.0004 |
| MEL | 61 | 0.0708 |
| NV | 178 | 0.1640 |
| SCCKA | 124 | 0.1180 |
| VASC | 7 | 0.0058 |

- mean_confidence: 0.805345356464386
- median_confidence: 0.8717259168624878
- mean_top1_top2_gap: 0.665729820728302
- mean_entropy: 0.5270761847496033
- low_confidence_rows: 89

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 30 | 0.060 |
| MEL | NV | 30 | 0.333 |
| BKL | BCC | 22 | 0.202 |
| BCC | BKL | 20 | 0.040 |
| BCC | AKIEC | 19 | 0.038 |
| AKIEC | SCCKA | 18 | 0.295 |
| MEL | BKL | 15 | 0.167 |
| SCCKA | BCC | 13 | 0.137 |
| BKL | SCCKA | 12 | 0.110 |
| NV | MEL | 11 | 0.074 |
| SCCKA | AKIEC | 11 | 0.116 |
| SCCKA | BKL | 11 | 0.116 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 30 | 0.060 |
| BCC | BKL | 20 | 0.040 |
| BCC | AKIEC | 19 | 0.038 |
| AKIEC | SCCKA | 18 | 0.295 |
| BKL | SCCKA | 12 | 0.110 |
| SCCKA | AKIEC | 11 | 0.116 |
| SCCKA | BKL | 11 | 0.116 |
| INF | BCC | 5 | 0.500 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 69.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
