# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_adaptive_gate_meta_gated_concat_weighted_sampler
- backbone: tf_efficientnetv2_b2
- metadata_fusion: gated_concat
- image_fusion: adaptive_gate
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

- accuracy: 0.7204198473282443
- balanced_accuracy: 0.5764471211699009
- dice_macro: 0.5368430122589921
- f1_macro: 0.5368430122589921
- roc_auc_macro_ovr: 0.9227068668710404
- top2_accuracy: 0.8597328244274809
- top3_accuracy: 0.9274809160305344

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
| AKIEC | 61 | 0.381818 | 0.688525 | 0.931104 | 0.491228 | 0.942498 |
| BCC | 504 | 0.918415 | 0.781746 | 0.935662 | 0.844587 | 0.957086 |
| BEN_OTH | 9 | 0.25 | 0.333333 | 0.991338 | 0.285714 | 0.898086 |
| BKL | 109 | 0.53211 | 0.53211 | 0.945687 | 0.53211 | 0.885101 |
| DF | 10 | 0.6 | 0.9 | 0.99422 | 0.72 | 0.995954 |
| INF | 10 | 0.2 | 0.4 | 0.984586 | 0.266667 | 0.938439 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.722275 |
| MEL | 90 | 0.714286 | 0.5 | 0.981211 | 0.588235 | 0.926026 |
| NV | 149 | 0.745763 | 0.885906 | 0.949944 | 0.809816 | 0.969257 |
| SCCKA | 95 | 0.584906 | 0.652632 | 0.95383 | 0.616915 | 0.925211 |
| VASC | 9 | 0.857143 | 0.666667 | 0.999038 | 0.75 | 0.989841 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.722275 |
| INF | 10 | 0.2 | 0.4 | 0.984586 | 0.266667 | 0.938439 |
| BEN_OTH | 9 | 0.25 | 0.333333 | 0.991338 | 0.285714 | 0.898086 |
| AKIEC | 61 | 0.381818 | 0.688525 | 0.931104 | 0.491228 | 0.942498 |
| BKL | 109 | 0.53211 | 0.53211 | 0.945687 | 0.53211 | 0.885101 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 110 | 0.1111 |
| BCC | 429 | 0.3737 |
| BEN_OTH | 12 | 0.0131 |
| BKL | 109 | 0.1277 |
| DF | 15 | 0.0167 |
| INF | 20 | 0.0236 |
| MAL_OTH | 0 | 0.0005 |
| MEL | 63 | 0.0664 |
| NV | 177 | 0.1616 |
| SCCKA | 106 | 0.0988 |
| VASC | 7 | 0.0067 |

- mean_confidence: 0.8277767300605774
- median_confidence: 0.9199304580688477
- mean_top1_top2_gap: 0.715505838394165
- mean_entropy: 0.4773326516151428
- low_confidence_rows: 96

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 32 | 0.063 |
| MEL | NV | 31 | 0.344 |
| BCC | SCCKA | 28 | 0.056 |
| BCC | BKL | 23 | 0.046 |
| BKL | BCC | 16 | 0.147 |
| SCCKA | AKIEC | 16 | 0.168 |
| BKL | AKIEC | 14 | 0.128 |
| AKIEC | BKL | 10 | 0.164 |
| BCC | INF | 10 | 0.020 |
| SCCKA | BKL | 10 | 0.105 |
| BKL | SCCKA | 9 | 0.083 |
| BCC | MEL | 7 | 0.014 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 32 | 0.063 |
| BCC | SCCKA | 28 | 0.056 |
| BCC | BKL | 23 | 0.046 |
| SCCKA | AKIEC | 16 | 0.168 |
| SCCKA | BKL | 10 | 0.105 |
| BKL | SCCKA | 9 | 0.083 |
| AKIEC | SCCKA | 5 | 0.082 |
| INF | BCC | 1 | 0.100 |
| INF | NV | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 83.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
