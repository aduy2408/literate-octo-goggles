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

- accuracy: 0.6717557251908397
- balanced_accuracy: 0.529488345956037
- dice_macro: 0.4841810638537401
- f1_macro: 0.4841810638537401
- roc_auc_macro_ovr: 0.9013913558117866
- top2_accuracy: 0.8625954198473282
- top3_accuracy: 0.9360687022900763

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
| AKIEC | 61 | 0.331034 | 0.786885 | 0.901722 | 0.466019 | 0.939841 | 0.815205 |
| BCC | 504 | 0.866817 | 0.761905 | 0.891544 | 0.810982 | 0.934196 | 0.793973 |
| BEN_OTH | 9 | 0.3 | 0.333333 | 0.993263 | 0.315789 | 0.932521 | 0.497002 |
| BKL | 109 | 0.477778 | 0.394495 | 0.949947 | 0.432161 | 0.794257 | 0.68771 |
| DF | 10 | 0.454545 | 0.5 | 0.99422 | 0.47619 | 0.977938 | 0.798325 |
| INF | 10 | 0.444444 | 0.4 | 0.995183 | 0.421053 | 0.940848 | 0.481965 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.618069 |  |
| MEL | 90 | 0.57377 | 0.388889 | 0.97286 | 0.463576 | 0.905822 | 0.752245 |
| NV | 149 | 0.715976 | 0.812081 | 0.946607 | 0.761006 | 0.955745 | 0.886697 |
| SCCKA | 95 | 0.569892 | 0.557895 | 0.958027 | 0.56383 | 0.925907 | 0.823937 |
| VASC | 9 | 0.470588 | 0.888889 | 0.991338 | 0.615385 | 0.990161 | 0.792628 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.618069 |  |
| BEN_OTH | 9 | 0.3 | 0.333333 | 0.993263 | 0.315789 | 0.932521 | 0.497002 |
| INF | 10 | 0.444444 | 0.4 | 0.995183 | 0.421053 | 0.940848 | 0.481965 |
| BKL | 109 | 0.477778 | 0.394495 | 0.949947 | 0.432161 | 0.794257 | 0.68771 |
| MEL | 90 | 0.57377 | 0.388889 | 0.97286 | 0.463576 | 0.905822 | 0.752245 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 145 | 0.1391 |
| BCC | 443 | 0.3704 |
| BEN_OTH | 10 | 0.0119 |
| BKL | 90 | 0.1138 |
| DF | 11 | 0.0125 |
| INF | 9 | 0.0102 |
| MAL_OTH | 0 | 0.0029 |
| MEL | 61 | 0.0655 |
| NV | 169 | 0.1614 |
| SCCKA | 93 | 0.0969 |
| VASC | 17 | 0.0155 |

- mean_confidence: 0.7505911588668823
- median_confidence: 0.7941693663597107
- mean_top1_top2_gap: 0.5936927795410156
- mean_entropy: 0.6975350379943848
- low_confidence_rows: 139

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 54 | 0.107 |
| MEL | NV | 33 | 0.367 |
| BKL | BCC | 24 | 0.220 |
| BCC | BKL | 22 | 0.044 |
| SCCKA | AKIEC | 22 | 0.232 |
| BCC | SCCKA | 21 | 0.042 |
| BKL | AKIEC | 16 | 0.147 |
| SCCKA | BCC | 14 | 0.147 |
| NV | MEL | 13 | 0.087 |
| BKL | SCCKA | 11 | 0.101 |
| BKL | NV | 8 | 0.073 |
| MEL | BCC | 7 | 0.078 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 54 | 0.107 |
| BCC | BKL | 22 | 0.044 |
| SCCKA | AKIEC | 22 | 0.232 |
| BCC | SCCKA | 21 | 0.042 |
| BKL | SCCKA | 11 | 0.101 |
| SCCKA | BKL | 6 | 0.063 |
| AKIEC | SCCKA | 4 | 0.066 |
| INF | BCC | 3 | 0.300 |
| INF | BEN_OTH | 1 | 0.100 |
| INF | NV | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 97.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
