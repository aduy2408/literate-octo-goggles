# Split Summary

## Full

- rows: 5240
- real_rows: 5240
- synthetic_rows: 0
- ignore_metadata_rows: 0

| class | count | synthetic |
|---|---:|---:|
| AKIEC | 303 | 0 |
| BCC | 2522 | 0 |
| BEN_OTH | 44 | 0 |
| BKL | 544 | 0 |
| DF | 52 | 0 |
| INF | 50 | 0 |
| MAL_OTH | 9 | 0 |
| MEL | 450 | 0 |
| NV | 746 | 0 |
| SCCKA | 473 | 0 |
| VASC | 47 | 0 |

## Train

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

## Val

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

- synthetic_train_only: False
- balance_mode: hybrid
- effective_rows_per_epoch: 3314
- strong_augmentation_classes: ['BEN_OTH', 'DF', 'INF', 'VASC']

| class | original train | effective per epoch |
|---|---:|---:|
| AKIEC | 242 | 242 |
| BCC | 2018 | 895 |
| BEN_OTH | 35 | 100 |
| BKL | 435 | 435 |
| DF | 42 | 100 |
| INF | 40 | 100 |
| MAL_OTH | 7 | 7 |
| MEL | 360 | 360 |
| NV | 597 | 597 |
| SCCKA | 378 | 378 |
| VASC | 38 | 100 |
