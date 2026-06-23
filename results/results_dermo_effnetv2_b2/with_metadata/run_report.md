# Dermoscopic Run Report

- fold: None
- backbone: tf_efficientnetv2_b2
- metadata_mode: concat
- loss: ce_f1

## Metrics

- accuracy: 0.7333333333333333
- balanced_accuracy: 0.6473940503519396
- f1_macro: 0.6631885868071691
- dice_macro: 0.6631885868071691
- roc_auc_macro_ovr: 0.9368063806735545
- top3_accuracy: 0.9367965367965368

## Per class

```
  class  support  precision  recall_sensitivity  specificity       f1  auc_ovr
  AKIEC       61   0.391892            0.475410     0.958867 0.429630 0.919606
    BCC      505   0.861538            0.887129     0.889231 0.874146 0.950324
BEN_OTH       23   0.666667            0.434783     0.995583 0.526316 0.930327
    BKL      109   0.425926            0.422018     0.940727 0.423963 0.846221
     DF       30   0.730769            0.633333     0.993778 0.678571 0.970252
    INF       28   0.636364            0.750000     0.989352 0.688525 0.958708
MAL_OTH       11   1.000000            0.727273     1.000000 0.842105 0.906469
    MEL       90   0.661765            0.500000     0.978404 0.569620 0.930589
     NV      149   0.703030            0.778523     0.951292 0.738854 0.955762
  SCCKA       95   0.683544            0.568421     0.976415 0.620690 0.941102
   VASC       54   0.864407            0.944444     0.992734 0.902655 0.995509
```

## Top confusions

- MEL -> NV: 26 (28.9%)
- BKL -> BCC: 23 (21.1%)
- BCC -> BKL: 20 (4.0%)
- SCCKA -> BCC: 14 (14.7%)
- SCCKA -> BKL: 14 (14.7%)
- AKIEC -> BKL: 13 (21.3%)
- BCC -> AKIEC: 13 (2.6%)
- BKL -> AKIEC: 13 (11.9%)
- SCCKA -> AKIEC: 12 (12.6%)
- AKIEC -> BCC: 11 (18.0%)
- NV -> MEL: 11 (7.4%)
- BKL -> SCCKA: 10 (9.2%)

## Warnings
