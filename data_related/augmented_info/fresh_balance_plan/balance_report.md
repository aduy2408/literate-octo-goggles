# MILK10k Balance Report

- BCC static cap: 1119
- verified usable synthetic lesions: 0
- source cap: 3
- synthetic/real cap: 2.0x

## Warnings

- QC summary was not provided; no synthetic row is considered verified usable.
- Synthetic image root was not provided; materialization is disabled.
- MAL_OTH should use external/manual-reviewed data; do not scale SD from only a few source lesions.

## Augmentation plan

```
  class  real_count  raw_synthetic_inventory  capped_inventory  verified_usable  target_total  kept_real  final_inventory_total  final_verified_total  additional_needed_inventory  additional_needed_verified
  AKIEC         303                        0                 0                0           303        303                    303                   303                            0                           0
    BCC        2522                        0                 0                0          2522       1119                   1119                  1119                            0                           0
BEN_OTH          44                        0                 0                0           132         44                     44                    44                           88                          88
    BKL         544                        0                 0                0           544        544                    544                   544                            0                           0
     DF          52                        0                 0                0           150         52                     52                    52                           98                          98
    INF          50                        0                 0                0           150         50                     50                    50                          100                         100
MAL_OTH           9                        0                 0                0            27          9                      9                     9                           18                          18
    MEL         450                        0                 0                0           450        450                    450                   450                            0                           0
     NV         746                        0                 0                0           746        746                    746                   746                            0                           0
  SCCKA         473                        0                 0                0           473        473                    473                   473                            0                           0
   VASC          47                        0                 0                0           141         47                     47                    47                           94                          94
```

## Source diversity

```
  class  inventory  inventory_sources  selected  selected_sources  max_selected_per_source
  AKIEC          0                  0         0                 0                        0
    BCC          0                  0         0                 0                        0
BEN_OTH          0                  0         0                 0                        0
    BKL          0                  0         0                 0                        0
     DF          0                  0         0                 0                        0
    INF          0                  0         0                 0                        0
MAL_OTH          0                  0         0                 0                        0
    MEL          0                  0         0                 0                        0
     NV          0                  0         0                 0                        0
  SCCKA          0                  0         0                 0                        0
   VASC          0                  0         0                 0                        0
```

## Selection reasons

Series([], )
