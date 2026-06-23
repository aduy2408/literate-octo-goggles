# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: runs/effnetv2_masks_balanced_cew
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ce
- class_weight: False
- weighted_sampler: False
- balance_mode: hybrid
- balance_head_ratio: 2.0
- balance_tail_floor: 100
- balance_min_source_count: 20
- augmented_data_dir: None
- dermoscopic_mask_dir: /marimo/milk10k_train_masks
- min_dermoscopic_mask_ratio: 0.01
- augmented_classes: []
- augmented_max_per_class: 0
- freeze_metadata_head: False
- zero_augmented_metadata: False

## Final Metrics

- accuracy: 0.6669847328244275
- balanced_accuracy: 0.38406071920731183
- dice_macro: 0.3880284378608632
- f1_macro: 0.3880284378608632
- roc_auc_macro_ovr: 0.9067732887013021
- top2_accuracy: 0.8549618320610687
- top3_accuracy: 0.9236641221374046

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
| AKIEC | 61 | 0.733333 | 0.180328 | 0.995947 | 0.289474 | 0.900078 |
| BCC | 504 | 0.842437 | 0.795635 | 0.862132 | 0.818367 | 0.908595 |
| BEN_OTH | 9 | 0 | 0 | 0.999038 | 0 | 0.957438 |
| BKL | 109 | 0.373016 | 0.431193 | 0.915868 | 0.4 | 0.79369 |
| DF | 10 | 0.75 | 0.3 | 0.999037 | 0.428571 | 0.948266 |
| INF | 10 | 0 | 0 | 0.998073 | 0 | 0.869268 |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.935946 |
| MEL | 90 | 0.6 | 0.3 | 0.981211 | 0.4 | 0.883519 |
| NV | 149 | 0.656566 | 0.872483 | 0.92436 | 0.74928 | 0.96554 |
| SCCKA | 95 | 0.431034 | 0.789474 | 0.896118 | 0.557621 | 0.935682 |
| VASC | 9 | 0.714286 | 0.555556 | 0.998075 | 0.625 | 0.876484 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 1 | 0 | 0.935946 |
| BEN_OTH | 9 | 0 | 0 | 0.999038 | 0 | 0.957438 |
| INF | 10 | 0 | 0 | 0.998073 | 0 | 0.869268 |
| AKIEC | 61 | 0.733333 | 0.180328 | 0.995947 | 0.289474 | 0.900078 |
| MEL | 90 | 0.6 | 0.3 | 0.981211 | 0.4 | 0.883519 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 15 | 0.0417 |
| BCC | 476 | 0.3935 |
| BEN_OTH | 1 | 0.0058 |
| BKL | 126 | 0.1474 |
| DF | 4 | 0.0049 |
| INF | 2 | 0.0052 |
| MAL_OTH | 0 | 0.0009 |
| MEL | 45 | 0.0536 |
| NV | 198 | 0.1815 |
| SCCKA | 174 | 0.1579 |
| VASC | 7 | 0.0074 |

- mean_confidence: 0.7641982436180115
- median_confidence: 0.8153038620948792
- mean_top1_top2_gap: 0.6213601231575012
- mean_entropy: 0.6840124726295471
- low_confidence_rows: 139

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 61 | 0.121 |
| MEL | NV | 41 | 0.456 |
| BKL | BCC | 29 | 0.266 |
| BCC | BKL | 28 | 0.056 |
| AKIEC | BKL | 22 | 0.361 |
| AKIEC | SCCKA | 16 | 0.262 |
| BKL | SCCKA | 15 | 0.138 |
| SCCKA | BCC | 15 | 0.158 |
| MEL | BKL | 13 | 0.144 |
| AKIEC | BCC | 11 | 0.180 |
| BKL | NV | 11 | 0.101 |
| NV | BKL | 11 | 0.074 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | SCCKA | 61 | 0.121 |
| BCC | BKL | 28 | 0.056 |
| AKIEC | SCCKA | 16 | 0.262 |
| BKL | SCCKA | 15 | 0.138 |
| INF | BCC | 5 | 0.500 |
| SCCKA | BKL | 4 | 0.042 |
| BCC | AKIEC | 1 | 0.002 |
| SCCKA | AKIEC | 1 | 0.011 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 90.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
