# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/single_encoder_split_runs/singleenc_canvas_meta_clean
- backbone: efficientnet_b2
- metadata_fusion: concat
- image_fusion: single_encoder_canvas
- loss: ce
- class_weight: False
- weighted_sampler: False
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

- accuracy: 0.732824427480916
- balanced_accuracy: 0.4447262538006909
- dice_macro: 0.4763148808801912
- f1_macro: 0.4763148808801912
- roc_auc_macro_ovr: 0.90824548970492
- top2_accuracy: 0.8826335877862596
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

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AKIEC | 61 | 0.533333 | 0.52459 | 0.971631 | 0.528926 | 0.931569 | 0.751253 |
| BCC | 504 | 0.830631 | 0.914683 | 0.827206 | 0.870633 | 0.944981 | 0.934259 |
| BEN_OTH | 9 | 0.25 | 0.222222 | 0.994225 | 0.235294 | 0.914768 | 0.582505 |
| BKL | 109 | 0.511364 | 0.412844 | 0.954207 | 0.456853 | 0.847554 | 0.823296 |
| DF | 10 | 1 | 0.4 | 1 | 0.571429 | 0.945087 | 0.632219 |
| INF | 10 | 0.5 | 0.1 | 0.999037 | 0.166667 | 0.951252 | 0.564142 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.666826 |  |
| MEL | 90 | 0.637681 | 0.488889 | 0.973904 | 0.553459 | 0.91778 | 0.800263 |
| NV | 149 | 0.745342 | 0.805369 | 0.954394 | 0.774194 | 0.966667 | 0.924132 |
| SCCKA | 95 | 0.585106 | 0.578947 | 0.959077 | 0.582011 | 0.928812 | 0.863229 |
| VASC | 9 | 0.571429 | 0.444444 | 0.997113 | 0.5 | 0.975404 | 0.887765 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.666826 |  |
| INF | 10 | 0.5 | 0.1 | 0.999037 | 0.166667 | 0.951252 | 0.564142 |
| BEN_OTH | 9 | 0.25 | 0.222222 | 0.994225 | 0.235294 | 0.914768 | 0.582505 |
| BKL | 109 | 0.511364 | 0.412844 | 0.954207 | 0.456853 | 0.847554 | 0.823296 |
| VASC | 9 | 0.571429 | 0.444444 | 0.997113 | 0.5 | 0.975404 | 0.887765 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 60 | 0.0587 |
| BCC | 555 | 0.5029 |
| BEN_OTH | 8 | 0.0075 |
| BKL | 88 | 0.1009 |
| DF | 4 | 0.0052 |
| INF | 2 | 0.0037 |
| MAL_OTH | 0 | 0.0007 |
| MEL | 69 | 0.0668 |
| NV | 161 | 0.1539 |
| SCCKA | 94 | 0.0935 |
| VASC | 7 | 0.0063 |

- mean_confidence: 0.8587908148765564
- median_confidence: 0.9537043571472168
- mean_top1_top2_gap: 0.7610636353492737
- mean_entropy: 0.3944314420223236
- low_confidence_rows: 54

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 31 | 0.284 |
| MEL | NV | 25 | 0.278 |
| SCCKA | BCC | 24 | 0.253 |
| AKIEC | SCCKA | 14 | 0.230 |
| NV | MEL | 13 | 0.087 |
| BKL | SCCKA | 12 | 0.110 |
| BCC | AKIEC | 11 | 0.022 |
| BCC | SCCKA | 11 | 0.022 |
| NV | BKL | 11 | 0.074 |
| AKIEC | BCC | 9 | 0.148 |
| BCC | BKL | 9 | 0.018 |
| MEL | BCC | 9 | 0.100 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| AKIEC | SCCKA | 14 | 0.230 |
| BKL | SCCKA | 12 | 0.110 |
| BCC | AKIEC | 11 | 0.022 |
| BCC | SCCKA | 11 | 0.022 |
| BCC | BKL | 9 | 0.018 |
| SCCKA | AKIEC | 8 | 0.084 |
| SCCKA | BKL | 8 | 0.084 |
| INF | BCC | 5 | 0.500 |
| INF | NV | 3 | 0.300 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
