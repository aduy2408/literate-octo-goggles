# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/single_encoder_split_runs/singleenc_canvas_meta_sampler05
- backbone: efficientnet_b2
- metadata_fusion: concat
- image_fusion: single_encoder_canvas
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

- accuracy: 0.6774809160305344
- balanced_accuracy: 0.4665233342016087
- dice_macro: 0.47014667387149883
- f1_macro: 0.47014667387149883
- roc_auc_macro_ovr: 0.9046510879118158
- top2_accuracy: 0.8645038167938931
- top3_accuracy: 0.9312977099236641

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
| AKIEC | 61 | 0.439394 | 0.47541 | 0.962513 | 0.456693 | 0.899131 | 0.719639 |
| BCC | 504 | 0.824701 | 0.821429 | 0.838235 | 0.823062 | 0.916017 | 0.846007 |
| BEN_OTH | 9 | 0.444444 | 0.444444 | 0.995188 | 0.444444 | 0.917977 | 0.56121 |
| BKL | 109 | 0.428571 | 0.330275 | 0.948882 | 0.373057 | 0.82945 | 0.750065 |
| DF | 10 | 0.5 | 0.3 | 0.99711 | 0.375 | 0.963198 | 0.899172 |
| INF | 10 | 0.2 | 0.1 | 0.996146 | 0.133333 | 0.934971 | 0.950753 |
| MAL_OTH | 2 | 0 | 0 | 0.999044 | 0 | 0.716539 |  |
| MEL | 90 | 0.552632 | 0.466667 | 0.964509 | 0.506024 | 0.907782 | 0.7485 |
| NV | 149 | 0.721854 | 0.731544 | 0.953281 | 0.726667 | 0.952699 | 0.893575 |
| SCCKA | 95 | 0.467626 | 0.684211 | 0.92235 | 0.555556 | 0.929652 | 0.85163 |
| VASC | 9 | 0.777778 | 0.777778 | 0.998075 | 0.777778 | 0.983745 | 0.778921 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 0.999044 | 0 | 0.716539 |  |
| INF | 10 | 0.2 | 0.1 | 0.996146 | 0.133333 | 0.934971 | 0.950753 |
| BKL | 109 | 0.428571 | 0.330275 | 0.948882 | 0.373057 | 0.82945 | 0.750065 |
| DF | 10 | 0.5 | 0.3 | 0.99711 | 0.375 | 0.963198 | 0.899172 |
| BEN_OTH | 9 | 0.444444 | 0.444444 | 0.995188 | 0.444444 | 0.917977 | 0.56121 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 66 | 0.0770 |
| BCC | 502 | 0.4313 |
| BEN_OTH | 9 | 0.0100 |
| BKL | 84 | 0.1108 |
| DF | 6 | 0.0065 |
| INF | 5 | 0.0056 |
| MAL_OTH | 1 | 0.0018 |
| MEL | 76 | 0.0714 |
| NV | 151 | 0.1461 |
| SCCKA | 139 | 0.1299 |
| VASC | 9 | 0.0097 |

- mean_confidence: 0.7871461510658264
- median_confidence: 0.8483368158340454
- mean_top1_top2_gap: 0.6446683406829834
- mean_entropy: 0.5853668451309204
- low_confidence_rows: 111

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 43 | 0.085 |
| BKL | BCC | 36 | 0.330 |
| MEL | NV | 25 | 0.278 |
| BCC | AKIEC | 20 | 0.040 |
| SCCKA | BCC | 20 | 0.211 |
| NV | MEL | 19 | 0.128 |
| AKIEC | SCCKA | 17 | 0.279 |
| NV | BKL | 13 | 0.087 |
| BCC | BKL | 12 | 0.024 |
| BKL | SCCKA | 11 | 0.101 |
| BKL | AKIEC | 10 | 0.092 |
| MEL | BCC | 10 | 0.111 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 43 | 0.085 |
| BCC | AKIEC | 20 | 0.040 |
| AKIEC | SCCKA | 17 | 0.279 |
| BCC | BKL | 12 | 0.024 |
| BKL | SCCKA | 11 | 0.101 |
| SCCKA | AKIEC | 5 | 0.053 |
| SCCKA | BKL | 5 | 0.053 |
| INF | BCC | 4 | 0.400 |
| INF | NV | 3 | 0.300 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 75.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
