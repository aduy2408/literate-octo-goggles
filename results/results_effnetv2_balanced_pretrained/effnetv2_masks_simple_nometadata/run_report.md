# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_masks_simple_nometadata
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ce
- class_weight: True
- weighted_sampler: False
- balance_mode: none
- balance_head_ratio: 2.0
- balance_tail_floor: 100
- balance_min_source_count: 20
- augmented_data_dir: None
- dermoscopic_mask_dir: /marimo/milk10k_train_masks
- min_dermoscopic_mask_ratio: 0.01
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.6536259541984732
- balanced_accuracy: 0.4780263108680734
- dice_macro: 0.4454925865556152
- f1_macro: 0.4454925865556152
- roc_auc_macro_ovr: 0.9183961160756415
- top2_accuracy: 0.8311068702290076
- top3_accuracy: 0.9179389312977099

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
| AKIEC | 61 | 0.473684 | 0.442623 | 0.969605 | 0.457627 | 0.900261 |
| BCC | 504 | 0.863326 | 0.751984 | 0.889706 | 0.803818 | 0.922747 |
| BEN_OTH | 9 | 0.454545 | 0.555556 | 0.994225 | 0.5 | 0.924607 |
| BKL | 109 | 0.333333 | 0.449541 | 0.895634 | 0.382812 | 0.804809 |
| DF | 10 | 0.375 | 0.6 | 0.990366 | 0.461538 | 0.976108 |
| INF | 10 | 0.0909091 | 0.1 | 0.990366 | 0.0952381 | 0.917341 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.922084 |
| MEL | 90 | 0.64 | 0.355556 | 0.981211 | 0.457143 | 0.879088 |
| NV | 149 | 0.591133 | 0.805369 | 0.907675 | 0.681818 | 0.942165 |
| SCCKA | 95 | 0.61 | 0.642105 | 0.959077 | 0.625641 | 0.940631 |
| VASC | 9 | 0.357143 | 0.555556 | 0.991338 | 0.434783 | 0.972516 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.922084 |
| INF | 10 | 0.0909091 | 0.1 | 0.990366 | 0.0952381 | 0.917341 |
| BKL | 109 | 0.333333 | 0.449541 | 0.895634 | 0.382812 | 0.804809 |
| VASC | 9 | 0.357143 | 0.555556 | 0.991338 | 0.434783 | 0.972516 |
| MEL | 90 | 0.64 | 0.355556 | 0.981211 | 0.457143 | 0.879088 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 57 | 0.0664 |
| BCC | 439 | 0.3611 |
| BEN_OTH | 11 | 0.0147 |
| BKL | 147 | 0.1644 |
| DF | 16 | 0.0178 |
| INF | 11 | 0.0138 |
| MAL_OTH | 0 | 0.0007 |
| MEL | 50 | 0.0522 |
| NV | 203 | 0.1930 |
| SCCKA | 100 | 0.1016 |
| VASC | 14 | 0.0143 |

- mean_confidence: 0.7737337946891785
- median_confidence: 0.8308212757110596
- mean_top1_top2_gap: 0.6356666684150696
- mean_entropy: 0.6465857028961182
- low_confidence_rows: 133

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 42 | 0.083 |
| MEL | NV | 37 | 0.411 |
| BKL | BCC | 25 | 0.229 |
| BCC | NV | 23 | 0.046 |
| BCC | SCCKA | 22 | 0.044 |
| AKIEC | BKL | 20 | 0.328 |
| BKL | NV | 18 | 0.165 |
| SCCKA | BCC | 15 | 0.158 |
| BCC | AKIEC | 14 | 0.028 |
| NV | BKL | 14 | 0.094 |
| SCCKA | BKL | 10 | 0.105 |
| BKL | SCCKA | 9 | 0.083 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 42 | 0.083 |
| BCC | SCCKA | 22 | 0.044 |
| BCC | AKIEC | 14 | 0.028 |
| SCCKA | BKL | 10 | 0.105 |
| BKL | SCCKA | 9 | 0.083 |
| AKIEC | SCCKA | 8 | 0.131 |
| SCCKA | AKIEC | 8 | 0.084 |
| INF | BCC | 4 | 0.400 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 78.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
