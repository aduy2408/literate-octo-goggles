# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/dual_encoder_ablation_runs_v2/run_baseline_ema
- backbone: efficientnet_b2
- metadata_fusion: concat
- image_fusion: concat
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

- accuracy: 0.7652671755725191
- balanced_accuracy: 0.5404660727221748
- dice_macro: 0.5435397203714502
- f1_macro: 0.5435397203714502
- roc_auc_macro_ovr: 0.9390122310184738
- top2_accuracy: 0.8893129770992366
- top3_accuracy: 0.9379770992366412

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
| AKIEC | 61 | 0.627451 | 0.52459 | 0.98075 | 0.571429 | 0.937117 | 0.660055 |
| BCC | 504 | 0.851852 | 0.912698 | 0.852941 | 0.881226 | 0.949722 | 0.819393 |
| BEN_OTH | 9 | 0.25 | 0.111111 | 0.997113 | 0.153846 | 0.931879 | 0.472425 |
| BKL | 109 | 0.695652 | 0.440367 | 0.977636 | 0.539326 | 0.876689 | 0.766033 |
| DF | 10 | 0.7 | 0.7 | 0.99711 | 0.7 | 0.993545 | 0.827338 |
| INF | 10 | 0.333333 | 0.4 | 0.992293 | 0.363636 | 0.931888 | 0.513106 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.863767 |  |
| MEL | 90 | 0.685714 | 0.533333 | 0.977035 | 0.6 | 0.939504 | 0.855946 |
| NV | 149 | 0.786982 | 0.892617 | 0.959956 | 0.836478 | 0.979246 | 0.918261 |
| SCCKA | 95 | 0.548673 | 0.652632 | 0.946485 | 0.596154 | 0.935616 | 0.748633 |
| VASC | 9 | 0.7 | 0.777778 | 0.997113 | 0.736842 | 0.990161 | 0.897798 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.863767 |  |
| BEN_OTH | 9 | 0.25 | 0.111111 | 0.997113 | 0.153846 | 0.931879 | 0.472425 |
| INF | 10 | 0.333333 | 0.4 | 0.992293 | 0.363636 | 0.931888 | 0.513106 |
| BKL | 109 | 0.695652 | 0.440367 | 0.977636 | 0.539326 | 0.876689 | 0.766033 |
| AKIEC | 61 | 0.627451 | 0.52459 | 0.98075 | 0.571429 | 0.937117 | 0.660055 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 51 | 0.0671 |
| BCC | 540 | 0.4382 |
| BEN_OTH | 4 | 0.0146 |
| BKL | 69 | 0.1032 |
| DF | 10 | 0.0134 |
| INF | 12 | 0.0160 |
| MAL_OTH | 0 | 0.0018 |
| MEL | 70 | 0.0762 |
| NV | 169 | 0.1542 |
| SCCKA | 113 | 0.1045 |
| VASC | 10 | 0.0108 |

- mean_confidence: 0.7705378532409668
- median_confidence: 0.835106611251831
- mean_top1_top2_gap: 0.6440498232841492
- mean_entropy: 0.7053409814834595
- low_confidence_rows: 149

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 28 | 0.257 |
| MEL | NV | 27 | 0.300 |
| BCC | SCCKA | 20 | 0.040 |
| SCCKA | BCC | 20 | 0.211 |
| AKIEC | SCCKA | 15 | 0.246 |
| BKL | SCCKA | 13 | 0.119 |
| AKIEC | BCC | 11 | 0.180 |
| MEL | BCC | 8 | 0.089 |
| NV | MEL | 7 | 0.047 |
| SCCKA | BKL | 7 | 0.074 |
| BCC | AKIEC | 6 | 0.012 |
| BCC | BKL | 6 | 0.012 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 20 | 0.040 |
| AKIEC | SCCKA | 15 | 0.246 |
| BKL | SCCKA | 13 | 0.119 |
| SCCKA | BKL | 7 | 0.074 |
| BCC | AKIEC | 6 | 0.012 |
| BCC | BKL | 6 | 0.012 |
| SCCKA | AKIEC | 5 | 0.053 |
| INF | BCC | 4 | 0.400 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
