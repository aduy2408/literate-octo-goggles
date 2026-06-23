# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_hybrid_tail_ablation/report_effv2b2_meta_concat_focal_hybrid_clean
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: focal
- class_weight: False
- weighted_sampler: False
- balance_mode: hybrid
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

- accuracy: 0.7786259541984732
- balanced_accuracy: 0.5436111405101479
- dice_macro: 0.5667102582869705
- f1_macro: 0.5667102582869705
- roc_auc_macro_ovr: 0.9182730296160215
- top2_accuracy: 0.9026717557251909
- top3_accuracy: 0.9475190839694656

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
| AKIEC | 61 | 0.622222 | 0.459016 | 0.982776 | 0.528302 | 0.944442 | 0.577422 |
| BCC | 504 | 0.867173 | 0.906746 | 0.871324 | 0.886518 | 0.956404 | 0.754219 |
| BEN_OTH | 9 | 0.333333 | 0.111111 | 0.998075 | 0.166667 | 0.813068 | 0.341654 |
| BKL | 109 | 0.72619 | 0.559633 | 0.975506 | 0.632124 | 0.872595 | 0.662214 |
| DF | 10 | 0.8 | 0.8 | 0.998073 | 0.8 | 0.938439 | 0.78386 |
| INF | 10 | 0.375 | 0.3 | 0.995183 | 0.333333 | 0.937765 | 0.487581 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.830784 |  |
| MEL | 90 | 0.758065 | 0.522222 | 0.984342 | 0.618421 | 0.93737 | 0.81901 |
| NV | 149 | 0.767442 | 0.885906 | 0.955506 | 0.82243 | 0.972945 | 0.834243 |
| SCCKA | 95 | 0.557252 | 0.768421 | 0.93914 | 0.646018 | 0.937936 | 0.726689 |
| VASC | 9 | 1 | 0.666667 | 1 | 0.8 | 0.959256 | 0.792376 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.830784 |  |
| BEN_OTH | 9 | 0.333333 | 0.111111 | 0.998075 | 0.166667 | 0.813068 | 0.341654 |
| INF | 10 | 0.375 | 0.3 | 0.995183 | 0.333333 | 0.937765 | 0.487581 |
| AKIEC | 61 | 0.622222 | 0.459016 | 0.982776 | 0.528302 | 0.944442 | 0.577422 |
| MEL | 90 | 0.758065 | 0.522222 | 0.984342 | 0.618421 | 0.93737 | 0.81901 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 45 | 0.0704 |
| BCC | 527 | 0.4056 |
| BEN_OTH | 3 | 0.0124 |
| BKL | 84 | 0.1249 |
| DF | 10 | 0.0122 |
| INF | 8 | 0.0175 |
| MAL_OTH | 0 | 0.0006 |
| MEL | 62 | 0.0737 |
| NV | 172 | 0.1488 |
| SCCKA | 131 | 0.1274 |
| VASC | 6 | 0.0066 |

- mean_confidence: 0.7161911725997925
- median_confidence: 0.7591204047203064
- mean_top1_top2_gap: 0.5524781346321106
- mean_entropy: 0.8295668959617615
- low_confidence_rows: 155

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| MEL | NV | 30 | 0.333 |
| BKL | BCC | 24 | 0.220 |
| BCC | SCCKA | 23 | 0.046 |
| AKIEC | SCCKA | 18 | 0.295 |
| BKL | SCCKA | 15 | 0.138 |
| SCCKA | BCC | 15 | 0.158 |
| AKIEC | BCC | 9 | 0.148 |
| MEL | BCC | 7 | 0.078 |
| BCC | AKIEC | 6 | 0.012 |
| BCC | MEL | 6 | 0.012 |
| INF | BCC | 6 | 0.600 |
| NV | BKL | 6 | 0.040 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 23 | 0.046 |
| AKIEC | SCCKA | 18 | 0.295 |
| BKL | SCCKA | 15 | 0.138 |
| BCC | AKIEC | 6 | 0.012 |
| INF | BCC | 6 | 0.600 |
| SCCKA | AKIEC | 6 | 0.063 |
| BCC | BKL | 5 | 0.010 |
| SCCKA | BKL | 1 | 0.011 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
