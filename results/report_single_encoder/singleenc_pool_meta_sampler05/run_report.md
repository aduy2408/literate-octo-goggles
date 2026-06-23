# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/single_encoder_split_runs/singleenc_pool_meta_sampler05
- backbone: efficientnet_b2
- metadata_fusion: concat
- image_fusion: shared_encoder_pool
- loss: ce
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

- accuracy: 0.6641221374045801
- balanced_accuracy: 0.4328933168114638
- dice_macro: 0.4380871905392807
- f1_macro: 0.4380871905392807
- roc_auc_macro_ovr: 0.8857308386990298
- top2_accuracy: 0.8530534351145038
- top3_accuracy: 0.9188931297709924

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
| AKIEC | 61 | 0.533333 | 0.262295 | 0.985816 | 0.351648 | 0.88988 | 0.661775 |
| BCC | 504 | 0.788462 | 0.813492 | 0.797794 | 0.800781 | 0.892843 | 0.801267 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.886964 | 0.612337 |
| BKL | 109 | 0.42 | 0.385321 | 0.938232 | 0.401914 | 0.759797 | 0.695573 |
| DF | 10 | 0.571429 | 0.4 | 0.99711 | 0.470588 | 0.972929 | 0.660877 |
| INF | 10 | 0.3 | 0.3 | 0.993256 | 0.3 | 0.916281 | 0.568308 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.703633 |  |
| MEL | 90 | 0.546875 | 0.388889 | 0.969729 | 0.454545 | 0.884482 | 0.761754 |
| NV | 149 | 0.7375 | 0.791946 | 0.953281 | 0.763754 | 0.960829 | 0.843852 |
| SCCKA | 95 | 0.435714 | 0.642105 | 0.917104 | 0.519149 | 0.897432 | 0.819874 |
| VASC | 9 | 0.6 | 0.666667 | 0.99615 | 0.631579 | 0.97797 | 0.790939 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.703633 |  |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.886964 | 0.612337 |
| INF | 10 | 0.3 | 0.3 | 0.993256 | 0.3 | 0.916281 | 0.568308 |
| AKIEC | 61 | 0.533333 | 0.262295 | 0.985816 | 0.351648 | 0.88988 | 0.661775 |
| BKL | 109 | 0.42 | 0.385321 | 0.938232 | 0.401914 | 0.759797 | 0.695573 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 30 | 0.0477 |
| BCC | 520 | 0.4243 |
| BEN_OTH | 7 | 0.0106 |
| BKL | 100 | 0.1232 |
| DF | 7 | 0.0098 |
| INF | 10 | 0.0119 |
| MAL_OTH | 0 | 0.0010 |
| MEL | 64 | 0.0728 |
| NV | 160 | 0.1459 |
| SCCKA | 140 | 0.1413 |
| VASC | 10 | 0.0115 |

- mean_confidence: 0.7500953674316406
- median_confidence: 0.7891206741333008
- mean_top1_top2_gap: 0.5957076549530029
- mean_entropy: 0.7045638561248779
- low_confidence_rows: 152

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 52 | 0.103 |
| BKL | BCC | 40 | 0.367 |
| MEL | NV | 29 | 0.322 |
| AKIEC | BCC | 23 | 0.377 |
| BCC | BKL | 21 | 0.042 |
| SCCKA | BCC | 21 | 0.221 |
| NV | MEL | 16 | 0.107 |
| AKIEC | SCCKA | 13 | 0.213 |
| BKL | SCCKA | 11 | 0.101 |
| MEL | BCC | 11 | 0.122 |
| MEL | BKL | 11 | 0.122 |
| AKIEC | BKL | 9 | 0.148 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 52 | 0.103 |
| BCC | BKL | 21 | 0.042 |
| AKIEC | SCCKA | 13 | 0.213 |
| BKL | SCCKA | 11 | 0.101 |
| SCCKA | BKL | 9 | 0.095 |
| BCC | AKIEC | 4 | 0.008 |
| INF | BCC | 4 | 0.400 |
| SCCKA | AKIEC | 3 | 0.032 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 77.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
