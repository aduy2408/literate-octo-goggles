# Dermoscopic Run Report

- fold: None
- backbone: tf_efficientnetv2_b2
- metadata_mode: none
- loss: ce_f1

## Metrics

- accuracy: 0.7012987012987013
- balanced_accuracy: 0.6260534639346079
- f1_macro: 0.6400293835642404
- dice_macro: 0.6400293835642404
- roc_auc_macro_ovr: 0.928844358099315
- top3_accuracy: 0.9264069264069265

## Per class

```
  class  support  precision  recall_sensitivity  specificity       f1  auc_ovr
  AKIEC       61   0.333333            0.540984     0.939671 0.412500 0.928612
    BCC      505   0.850000            0.875248     0.880000 0.862439 0.945633
BEN_OTH       23   0.733333            0.478261     0.996466 0.578947 0.942311
    BKL      109   0.380000            0.348624     0.940727 0.363636 0.813225
     DF       30   0.720000            0.600000     0.993778 0.654545 0.920504
    INF       28   0.709677            0.785714     0.992014 0.745763 0.980479
MAL_OTH       11   1.000000            0.727273     1.000000 0.842105 0.903687
    MEL       90   0.552239            0.411111     0.971831 0.471338 0.916766
     NV      149   0.650602            0.724832     0.942346 0.685714 0.949945
  SCCKA       95   0.650794            0.431579     0.979245 0.518987 0.918818
   VASC       54   0.852459            0.962963     0.991826 0.904348 0.997309
```

## Top confusions

- MEL -> NV: 28 (31.1%)
- BKL -> BCC: 24 (22.0%)
- BCC -> BKL: 20 (4.0%)
- SCCKA -> AKIEC: 20 (21.1%)
- BCC -> AKIEC: 18 (3.6%)
- BKL -> AKIEC: 18 (16.5%)
- SCCKA -> BCC: 18 (18.9%)
- BKL -> NV: 15 (13.8%)
- NV -> MEL: 15 (10.1%)
- SCCKA -> BKL: 13 (13.7%)
- AKIEC -> BKL: 11 (18.0%)
- NV -> BCC: 11 (7.4%)

## Warnings
