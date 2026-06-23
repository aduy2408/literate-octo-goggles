# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/dual_encoder_ablation_runs_v2/run_tau025_sampler_ema
- backbone: efficientnet_b2
- metadata_fusion: concat
- image_fusion: concat
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

- accuracy: 0.7538167938931297
- balanced_accuracy: 0.5756662759690905
- dice_macro: 0.5536300408403592
- f1_macro: 0.5536300408403592
- roc_auc_macro_ovr: 0.9188277203283789
- top2_accuracy: 0.8807251908396947
- top3_accuracy: 0.9351145038167938

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
| AKIEC | 61 | 0.443038 | 0.57377 | 0.95542 | 0.5 | 0.936519 | 0.687161 |
| BCC | 504 | 0.890496 | 0.855159 | 0.902574 | 0.87247 | 0.951823 | 0.812341 |
| BEN_OTH | 9 | 0.333333 | 0.333333 | 0.994225 | 0.333333 | 0.939899 | 0.420182 |
| BKL | 109 | 0.593407 | 0.495413 | 0.960596 | 0.54 | 0.85113 | 0.766896 |
| DF | 10 | 0.666667 | 0.8 | 0.996146 | 0.727273 | 0.989788 | 0.746172 |
| INF | 10 | 0.333333 | 0.3 | 0.99422 | 0.315789 | 0.930732 | 0.641938 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.660612 |  |
| MEL | 90 | 0.680851 | 0.711111 | 0.968685 | 0.695652 | 0.948875 | 0.869453 |
| NV | 149 | 0.883212 | 0.812081 | 0.982202 | 0.846154 | 0.978798 | 0.822643 |
| SCCKA | 95 | 0.528926 | 0.673684 | 0.940189 | 0.592593 | 0.941172 | 0.780016 |
| VASC | 9 | 0.583333 | 0.777778 | 0.995188 | 0.666667 | 0.977756 | 0.936962 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.660612 |  |
| INF | 10 | 0.333333 | 0.3 | 0.99422 | 0.315789 | 0.930732 | 0.641938 |
| BEN_OTH | 9 | 0.333333 | 0.333333 | 0.994225 | 0.333333 | 0.939899 | 0.420182 |
| AKIEC | 61 | 0.443038 | 0.57377 | 0.95542 | 0.5 | 0.936519 | 0.687161 |
| BKL | 109 | 0.593407 | 0.495413 | 0.960596 | 0.54 | 0.85113 | 0.766896 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 79 | 0.0888 |
| BCC | 484 | 0.3967 |
| BEN_OTH | 9 | 0.0138 |
| BKL | 91 | 0.1166 |
| DF | 12 | 0.0152 |
| INF | 9 | 0.0131 |
| MAL_OTH | 0 | 0.0025 |
| MEL | 94 | 0.1036 |
| NV | 137 | 0.1206 |
| SCCKA | 121 | 0.1136 |
| VASC | 12 | 0.0155 |

- mean_confidence: 0.7548145651817322
- median_confidence: 0.8110538125038147
- mean_top1_top2_gap: 0.6143200397491455
- mean_entropy: 0.7280750870704651
- low_confidence_rows: 161

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 25 | 0.050 |
| BKL | BCC | 22 | 0.202 |
| BCC | AKIEC | 19 | 0.038 |
| BCC | BKL | 18 | 0.036 |
| AKIEC | SCCKA | 17 | 0.279 |
| NV | MEL | 16 | 0.107 |
| BKL | SCCKA | 12 | 0.110 |
| MEL | NV | 12 | 0.133 |
| BKL | AKIEC | 11 | 0.101 |
| SCCKA | AKIEC | 11 | 0.116 |
| SCCKA | BCC | 10 | 0.105 |
| SCCKA | BKL | 8 | 0.084 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 25 | 0.050 |
| BCC | AKIEC | 19 | 0.038 |
| BCC | BKL | 18 | 0.036 |
| AKIEC | SCCKA | 17 | 0.279 |
| BKL | SCCKA | 12 | 0.110 |
| SCCKA | AKIEC | 11 | 0.116 |
| SCCKA | BKL | 8 | 0.084 |
| INF | BCC | 4 | 0.400 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 62.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
