# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine3_fusion_loss/report_effv2b2_meta_low_rank_bilinear_focal_s05_clean
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: low_rank_bilinear
- loss: focal
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

- accuracy: 0.7471374045801527
- balanced_accuracy: 0.5437922668561391
- dice_macro: 0.5439746243907099
- f1_macro: 0.5439746243907099
- roc_auc_macro_ovr: 0.9192076756224381
- top2_accuracy: 0.8902671755725191
- top3_accuracy: 0.9408396946564885

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
| AKIEC | 61 | 0.392857 | 0.721311 | 0.931104 | 0.508671 | 0.951717 |
| BCC | 504 | 0.890041 | 0.85119 | 0.902574 | 0.870183 | 0.953978 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.881938 |
| BKL | 109 | 0.566038 | 0.550459 | 0.951012 | 0.55814 | 0.864818 |
| DF | 10 | 0.727273 | 0.8 | 0.99711 | 0.761905 | 0.975915 |
| INF | 10 | 0.428571 | 0.3 | 0.996146 | 0.352941 | 0.945857 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.719407 |
| MEL | 90 | 0.688312 | 0.588889 | 0.974948 | 0.634731 | 0.944236 |
| NV | 149 | 0.796296 | 0.865772 | 0.963293 | 0.829582 | 0.976805 |
| SCCKA | 95 | 0.632911 | 0.526316 | 0.96957 | 0.574713 | 0.923455 |
| VASC | 9 | 0.857143 | 0.666667 | 0.999038 | 0.75 | 0.973158 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.719407 |
| BEN_OTH | 9 | 0.2 | 0.111111 | 0.99615 | 0.142857 | 0.881938 |
| INF | 10 | 0.428571 | 0.3 | 0.996146 | 0.352941 | 0.945857 |
| AKIEC | 61 | 0.392857 | 0.721311 | 0.931104 | 0.508671 | 0.951717 |
| BKL | 109 | 0.566038 | 0.550459 | 0.951012 | 0.55814 | 0.864818 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 112 | 0.1178 |
| BCC | 482 | 0.3568 |
| BEN_OTH | 5 | 0.0121 |
| BKL | 106 | 0.1531 |
| DF | 11 | 0.0169 |
| INF | 7 | 0.0103 |
| MAL_OTH | 0 | 0.0009 |
| MEL | 77 | 0.0927 |
| NV | 162 | 0.1372 |
| SCCKA | 79 | 0.0934 |
| VASC | 7 | 0.0087 |

- mean_confidence: 0.6879563331604004
- median_confidence: 0.7046300172805786
- mean_top1_top2_gap: 0.4925142824649811
- mean_entropy: 0.8405203223228455
- low_confidence_rows: 187

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 33 | 0.065 |
| MEL | NV | 23 | 0.256 |
| SCCKA | AKIEC | 20 | 0.211 |
| BKL | BCC | 18 | 0.165 |
| BCC | BKL | 17 | 0.034 |
| BCC | SCCKA | 14 | 0.028 |
| BKL | AKIEC | 14 | 0.128 |
| SCCKA | BKL | 12 | 0.126 |
| BKL | SCCKA | 9 | 0.083 |
| NV | MEL | 9 | 0.060 |
| SCCKA | BCC | 9 | 0.095 |
| MEL | BCC | 8 | 0.089 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 33 | 0.065 |
| SCCKA | AKIEC | 20 | 0.211 |
| BCC | BKL | 17 | 0.034 |
| BCC | SCCKA | 14 | 0.028 |
| SCCKA | BKL | 12 | 0.126 |
| BKL | SCCKA | 9 | 0.083 |
| AKIEC | SCCKA | 5 | 0.082 |
| INF | BCC | 3 | 0.300 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 64.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
