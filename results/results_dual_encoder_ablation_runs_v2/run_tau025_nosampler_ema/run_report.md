# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/dual_encoder_ablation_runs_v2/run_tau025_nosampler_ema
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

- accuracy: 0.7824427480916031
- balanced_accuracy: 0.5610085721202881
- dice_macro: 0.5497956155156963
- f1_macro: 0.5497956155156963
- roc_auc_macro_ovr: 0.9313132863970424
- top2_accuracy: 0.8940839694656488
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
| AKIEC | 61 | 0.538462 | 0.57377 | 0.969605 | 0.555556 | 0.942814 | 0.694162 |
| BCC | 504 | 0.867675 | 0.910714 | 0.871324 | 0.888674 | 0.953906 | 0.85281 |
| BEN_OTH | 9 | 0.25 | 0.111111 | 0.997113 | 0.153846 | 0.943856 | 0.435379 |
| BKL | 109 | 0.683544 | 0.495413 | 0.973376 | 0.574468 | 0.880968 | 0.785053 |
| DF | 10 | 0.666667 | 0.8 | 0.996146 | 0.727273 | 0.987958 | 0.75456 |
| INF | 10 | 0.333333 | 0.3 | 0.99422 | 0.315789 | 0.943449 | 0.500105 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.732314 |  |
| MEL | 90 | 0.722892 | 0.666667 | 0.975992 | 0.693642 | 0.949235 | 0.865149 |
| NV | 149 | 0.884354 | 0.872483 | 0.98109 | 0.878378 | 0.982344 | 0.856301 |
| SCCKA | 95 | 0.588785 | 0.663158 | 0.95383 | 0.623762 | 0.942575 | 0.780011 |
| VASC | 9 | 0.538462 | 0.777778 | 0.994225 | 0.636364 | 0.985028 | 0.959629 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr | mean_correct_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.732314 |  |
| BEN_OTH | 9 | 0.25 | 0.111111 | 0.997113 | 0.153846 | 0.943856 | 0.435379 |
| INF | 10 | 0.333333 | 0.3 | 0.99422 | 0.315789 | 0.943449 | 0.500105 |
| AKIEC | 61 | 0.538462 | 0.57377 | 0.969605 | 0.555556 | 0.942814 | 0.694162 |
| BKL | 109 | 0.683544 | 0.495413 | 0.973376 | 0.574468 | 0.880968 | 0.785053 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 65 | 0.0728 |
| BCC | 529 | 0.4448 |
| BEN_OTH | 4 | 0.0135 |
| BKL | 79 | 0.0987 |
| DF | 12 | 0.0155 |
| INF | 9 | 0.0137 |
| MAL_OTH | 0 | 0.0061 |
| MEL | 83 | 0.0917 |
| NV | 147 | 0.1318 |
| SCCKA | 107 | 0.0989 |
| VASC | 13 | 0.0123 |

- mean_confidence: 0.7887431383132935
- median_confidence: 0.867466390132904
- mean_top1_top2_gap: 0.6728351712226868
- mean_entropy: 0.670480489730835
- low_confidence_rows: 128

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BKL | BCC | 26 | 0.239 |
| BCC | SCCKA | 17 | 0.034 |
| SCCKA | BCC | 15 | 0.158 |
| AKIEC | SCCKA | 13 | 0.213 |
| MEL | NV | 12 | 0.133 |
| BKL | SCCKA | 11 | 0.101 |
| BCC | AKIEC | 10 | 0.020 |
| MEL | BCC | 10 | 0.111 |
| SCCKA | AKIEC | 10 | 0.105 |
| BKL | AKIEC | 9 | 0.083 |
| NV | MEL | 9 | 0.060 |
| AKIEC | BCC | 7 | 0.115 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 17 | 0.034 |
| AKIEC | SCCKA | 13 | 0.213 |
| BKL | SCCKA | 11 | 0.101 |
| BCC | AKIEC | 10 | 0.020 |
| SCCKA | AKIEC | 10 | 0.105 |
| BCC | BKL | 7 | 0.014 |
| SCCKA | BKL | 6 | 0.063 |
| INF | BCC | 4 | 0.400 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
