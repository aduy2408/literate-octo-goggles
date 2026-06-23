# MILK10k Balance Report

- BCC static cap: 1119
- verified usable synthetic lesions: 0
- source cap: 3
- synthetic/real cap: 2.0x

## Warnings

- QC summary was not provided; no synthetic row is considered verified usable.
- Synthetic image root was not provided; materialization is disabled.
- Synthetic images missing: 1064 files across 532 incomplete inventory pairs.
- MAL_OTH should use external/manual-reviewed data; do not scale SD from only a few source lesions.

## Augmentation plan

```
  class  real_count  raw_synthetic_inventory  capped_inventory  verified_usable  target_total  kept_real  final_inventory_total  final_verified_total  additional_needed_inventory  additional_needed_verified
  AKIEC         303                        0                 0                0           303        303                    303                   303                            0                           0
    BCC        2522                        0                 0                0          2522       1119                   1119                  1119                            0                           0
BEN_OTH          44                       71                60                0           132         44                    104                    44                           28                          88
    BKL         544                        0                 0                0           544        544                    544                   544                            0                           0
     DF          52                       97                79                0           150         52                    131                    52                           19                          98
    INF          50                       90                78                0           150         50                    128                    50                           22                         100
MAL_OTH           9                       49                18                0            27          9                     27                     9                            0                          18
    MEL         450                        0                 0                0           450        450                    450                   450                            0                           0
     NV         746                        0                 0                0           746        746                    746                   746                            0                           0
  SCCKA         473                        0                 0                0           473        473                    473                   473                            0                           0
   VASC          47                      225                94                0           141         47                    141                    47                            0                          94
```

## Source diversity

```
  class  inventory  inventory_sources  selected  selected_sources  max_selected_per_source
  AKIEC          0                  0         0                 0                        0
    BCC          0                  0         0                 0                        0
BEN_OTH         71                 29        60                29                        3
    BKL          0                  0         0                 0                        0
     DF         97                 35        79                35                        3
    INF         90                 35        78                35                        3
MAL_OTH         49                  8        18                 7                        3
    MEL          0                  0         0                 0                        0
     NV          0                  0         0                 0                        0
  SCCKA          0                  0         0                 0                        0
   VASC        225                 45        94                33                        3
```

## Selection reasons

selection_reason
missing_image_pair    532
