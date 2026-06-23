# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/milk10k_effv2b2_machine2_no_metadata_compact8/effv2b2_ce_sampler05_no_metadata_concat_simple
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ce
- class_weight: False
- weighted_sampler: True
- augmented_data_dir: None
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.7395038167938931
- balanced_accuracy: 0.5397399427472007
- dice_macro: 0.5384276738048196
- f1_macro: 0.5384276738048196
- roc_auc_macro_ovr: 0.9121452839123374
- top2_accuracy: 0.8845419847328244
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

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| AKIEC | 61 | 0.444444 | 0.721311 | 0.944276 | 0.55 | 0.946634 |
| BCC | 504 | 0.914097 | 0.823413 | 0.928309 | 0.866388 | 0.960868 |
| BEN_OTH | 9 | 0.222222 | 0.222222 | 0.993263 | 0.222222 | 0.894664 |
| BKL | 109 | 0.516129 | 0.587156 | 0.936102 | 0.549356 | 0.888638 |
| DF | 10 | 0.857143 | 0.6 | 0.999037 | 0.705882 | 0.979191 |
| INF | 10 | 0.428571 | 0.3 | 0.996146 | 0.352941 | 0.914258 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.629063 |
| MEL | 90 | 0.623529 | 0.588889 | 0.966597 | 0.605714 | 0.939225 |
| NV | 149 | 0.766467 | 0.85906 | 0.956618 | 0.810127 | 0.978022 |
| SCCKA | 95 | 0.62069 | 0.568421 | 0.965373 | 0.593407 | 0.919932 |
| VASC | 9 | 0.666667 | 0.666667 | 0.997113 | 0.666667 | 0.983103 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.629063 |
| BEN_OTH | 9 | 0.222222 | 0.222222 | 0.993263 | 0.222222 | 0.894664 |
| INF | 10 | 0.428571 | 0.3 | 0.996146 | 0.352941 | 0.914258 |
| BKL | 109 | 0.516129 | 0.587156 | 0.936102 | 0.549356 | 0.888638 |
| AKIEC | 61 | 0.444444 | 0.721311 | 0.944276 | 0.55 | 0.946634 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 99 | 0.1024 |
| BCC | 454 | 0.4012 |
| BEN_OTH | 9 | 0.0107 |
| BKL | 124 | 0.1349 |
| DF | 7 | 0.0070 |
| INF | 7 | 0.0079 |
| MAL_OTH | 0 | 0.0005 |
| MEL | 85 | 0.0850 |
| NV | 167 | 0.1573 |
| SCCKA | 87 | 0.0857 |
| VASC | 9 | 0.0073 |

- mean_confidence: 0.8331688046455383
- median_confidence: 0.9167246222496033
- mean_top1_top2_gap: 0.7211318612098694
- mean_entropy: 0.46819478273391724
- low_confidence_rows: 84

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 28 | 0.056 |
| MEL | NV | 26 | 0.289 |
| BCC | BKL | 21 | 0.042 |
| BCC | SCCKA | 19 | 0.038 |
| SCCKA | AKIEC | 17 | 0.179 |
| BKL | BCC | 15 | 0.138 |
| SCCKA | BKL | 15 | 0.158 |
| NV | MEL | 10 | 0.067 |
| BCC | MEL | 9 | 0.018 |
| BKL | AKIEC | 9 | 0.083 |
| BKL | SCCKA | 9 | 0.083 |
| AKIEC | BKL | 8 | 0.131 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | AKIEC | 28 | 0.056 |
| BCC | BKL | 21 | 0.042 |
| BCC | SCCKA | 19 | 0.038 |
| SCCKA | AKIEC | 17 | 0.179 |
| SCCKA | BKL | 15 | 0.158 |
| BKL | SCCKA | 9 | 0.083 |
| AKIEC | SCCKA | 3 | 0.049 |
| INF | BCC | 3 | 0.300 |
| INF | BEN_OTH | 1 | 0.100 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 68.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
