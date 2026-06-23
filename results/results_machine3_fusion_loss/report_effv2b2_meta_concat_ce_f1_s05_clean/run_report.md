# MILK10k Run Report

## Config Summary

- fold: None
- output_dir: /marimo/report_runs_machine3_fusion_loss/report_effv2b2_meta_concat_ce_f1_s05_clean
- backbone: tf_efficientnetv2_b2
- metadata_fusion: concat
- image_fusion: concat
- loss: ce_f1
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

- accuracy: 0.7633587786259542
- balanced_accuracy: 0.5724645658991084
- dice_macro: 0.563722093281272
- f1_macro: 0.563722093281272
- roc_auc_macro_ovr: 0.9345681825164571
- top2_accuracy: 0.8854961832061069
- top3_accuracy: 0.950381679389313

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
| AKIEC | 61 | 0.486486 | 0.590164 | 0.961499 | 0.533333 | 0.940323 |
| BCC | 504 | 0.918455 | 0.849206 | 0.930147 | 0.882474 | 0.959595 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.909849 |
| BKL | 109 | 0.561983 | 0.623853 | 0.943557 | 0.591304 | 0.870426 |
| DF | 10 | 0.615385 | 0.8 | 0.995183 | 0.695652 | 0.988632 |
| INF | 10 | 0.5 | 0.4 | 0.996146 | 0.444444 | 0.955973 |
| MAL_OTH | 2 | 0 | 0 | 0.999044 | 0 | 0.834608 |
| MEL | 90 | 0.732394 | 0.577778 | 0.980167 | 0.645963 | 0.945801 |
| NV | 149 | 0.792683 | 0.872483 | 0.96218 | 0.830671 | 0.975424 |
| SCCKA | 95 | 0.573913 | 0.694737 | 0.948583 | 0.628571 | 0.925819 |
| VASC | 9 | 0.875 | 0.777778 | 0.999038 | 0.823529 | 0.9738 |

## Weak Classes

| class | support | precision | recall_sensitivity | specificity | f1 | auc_ovr |
| --- | --- | --- | --- | --- | --- | --- |
| MAL_OTH | 2 | 0 | 0 | 0.999044 | 0 | 0.834608 |
| BEN_OTH | 9 | 0.142857 | 0.111111 | 0.994225 | 0.125 | 0.909849 |
| INF | 10 | 0.5 | 0.4 | 0.996146 | 0.444444 | 0.955973 |
| AKIEC | 61 | 0.486486 | 0.590164 | 0.961499 | 0.533333 | 0.940323 |
| BKL | 109 | 0.561983 | 0.623853 | 0.943557 | 0.591304 | 0.870426 |

## Prediction Distribution

| class | pred_count | mean_prob |
|---|---:|---:|
| AKIEC | 74 | 0.0867 |
| BCC | 466 | 0.4086 |
| BEN_OTH | 7 | 0.0095 |
| BKL | 121 | 0.1259 |
| DF | 13 | 0.0136 |
| INF | 8 | 0.0082 |
| MAL_OTH | 1 | 0.0005 |
| MEL | 71 | 0.0765 |
| NV | 164 | 0.1511 |
| SCCKA | 115 | 0.1111 |
| VASC | 8 | 0.0082 |

- mean_confidence: 0.8377542495727539
- median_confidence: 0.920281171798706
- mean_top1_top2_gap: 0.7283874154090881
- mean_entropy: 0.4567090570926666
- low_confidence_rows: 81

## Top Confusion Pairs

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 23 | 0.046 |
| MEL | NV | 23 | 0.256 |
| BCC | SCCKA | 21 | 0.042 |
| BCC | AKIEC | 17 | 0.034 |
| BKL | SCCKA | 14 | 0.128 |
| BKL | BCC | 13 | 0.119 |
| SCCKA | AKIEC | 11 | 0.116 |
| AKIEC | BKL | 10 | 0.164 |
| SCCKA | BKL | 10 | 0.105 |
| AKIEC | SCCKA | 9 | 0.148 |
| SCCKA | BCC | 8 | 0.084 |
| BKL | AKIEC | 6 | 0.055 |

## Watched Confusion Patterns

| true | predicted | count | rate_of_true |
|---|---|---:|---:|
| BCC | BKL | 23 | 0.046 |
| BCC | SCCKA | 21 | 0.042 |
| BCC | AKIEC | 17 | 0.034 |
| BKL | SCCKA | 14 | 0.128 |
| SCCKA | AKIEC | 11 | 0.116 |
| SCCKA | BKL | 10 | 0.105 |
| AKIEC | SCCKA | 9 | 0.148 |
| INF | BCC | 2 | 0.200 |

## Warnings

- [high] bcc_boundary_drift: BCC -> AKIEC/BKL/SCCKA count is 61.
- [medium] tiny_validation_support: MAL_OTH validation support is only 2.
