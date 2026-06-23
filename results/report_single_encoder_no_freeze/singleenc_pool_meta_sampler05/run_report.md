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

- accuracy: 0.7013358778625954
- balanced_accuracy: 0.456562920678067
- dice_macro: 0.4898087377057165
- f1_macro: 0.4898087377057165
- roc_auc_macro_ovr: 0.8960721407863715
- top2_accuracy: 0.8778625954198473
- top3_accuracy: 0.9341603053435115

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
| AKIEC | 61 | 0.423529 | 0.590164 | 0.950355 | 0.493151 | 0.92358 | 0.824212 |
| BCC | 504 | 0.841085 | 0.861111 | 0.849265 | 0.85098 | 0.926358 | 0.913965 |
| BEN_OTH | 9 | 0.25 | 0.111111 | 0.997113 | 0.153846 | 0.905785 | 0.834692 |
| BKL | 109 | 0.47191 | 0.385321 | 0.949947 | 0.424242 | 0.795488 | 0.832611 |
| DF | 10 | 1 | 0.4 | 1 | 0.571429 | 0.985742 | 0.683109 |
| INF | 10 | 0.666667 | 0.2 | 0.999037 | 0.307692 | 0.921869 | 0.806394 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.620459 |  |
| MEL | 90 | 0.512195 | 0.466667 | 0.958246 | 0.488372 | 0.904384 | 0.888874 |
| NV | 149 | 0.746667 | 0.751678 | 0.957731 | 0.749164 | 0.963935 | 0.930399 |
| SCCKA | 95 | 0.513761 | 0.589474 | 0.944386 | 0.54902 | 0.926625 | 0.898608 |
| VASC | 9 | 1 | 0.666667 | 1 | 0.8 | 0.982569 | 0.83998 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.620459 |  |
| BEN_OTH | 9 | 0.25 | 0.111111 | 0.997113 | 0.153846 | 0.905785 | 0.834692 |
| INF | 10 | 0.666667 | 0.2 | 0.999037 | 0.307692 | 0.921869 | 0.806394 |
| BKL | 109 | 0.47191 | 0.385321 | 0.949947 | 0.424242 | 0.795488 | 0.832611 |
| MEL | 90 | 0.512195 | 0.466667 | 0.958246 | 0.488372 | 0.904384 | 0.888874 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 85 | 0.0807 |
| BCC | 516 | 0.4676 |
| BEN_OTH | 4 | 0.0053 |
| BKL | 89 | 0.1016 |
| DF | 4 | 0.0039 |
| INF | 3 | 0.0037 |
| MAL_OTH | 0 | 0.0004 |
| MEL | 82 | 0.0777 |
| NV | 150 | 0.1417 |
| SCCKA | 109 | 0.1112 |
| VASC | 6 | 0.0063 |

- mean_confidence: 0.8618848323822021
- median_confidence: 0.9437005519866943
- mean_top1_top2_gap: 0.7573769092559814
- mean_entropy: 0.3766627907752991
- low_confidence_rows: 37

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 32 | 0.063 |
| BKL | BCC | 29 | 0.266 |
| MEL | NV | 25 | 0.278 |
| NV | MEL | 24 | 0.161 |
| BCC | AKIEC | 22 | 0.044 |
| SCCKA | BCC | 17 | 0.179 |
| BKL | AKIEC | 14 | 0.128 |
| MEL | BCC | 12 | 0.133 |
| SCCKA | BKL | 12 | 0.126 |
| BCC | BKL | 11 | 0.022 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | AKIEC | 10 | 0.105 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 32 | 0.063 |
| BCC | AKIEC | 22 | 0.044 |
| SCCKA | BKL | 12 | 0.126 |
| BCC | BKL | 11 | 0.022 |
| AKIEC | SCCKA | 10 | 0.164 |
| SCCKA | AKIEC | 10 | 0.105 |
| BKL | SCCKA | 8 | 0.073 |
| INF | BCC | 7 | 0.700 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 65.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
