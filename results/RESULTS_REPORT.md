# Báo cáo tổng hợp kết quả MILK10k

> Báo cáo này ưu tiên các rerun controlled mới trong `results/results_machine1_backbones`, `results/report_runs_machine2_meta_aug_derm`, và `results/results_machine3_fusion_loss`. Các run cũ vẫn được giữ dưới nhãn exploratory / legacy để tra artifact, nhưng không còn là phần kết luận chính.

## 1. Controlled Comparison Summary

### Kết luận chính

- Bộ rerun mới đã đủ để so backbone, metadata, augmentation, fusion và loss trên cùng validation split sạch.
- Validation của các run controlled dùng chung checksum `ad42f5...b645`, có `1048` mẫu real và `0` mẫu synthetic.
- Run augmentation source-safe tốt nhất hiện tại là `report_effv2b2_meta_concat_focal_s05_aug_inf15_safe` với macro-F1 `0.5828`.
- `dice_macro` trong các `metrics.json` hiện tại bằng đúng `f1_macro` cho các run controlled đã kiểm tra; báo cáo vẫn ghi riêng Dice macro ở bảng chính/loss để tiện theo dõi metric này.
- Trong EMA/LWS matrix, `run_tau050_sampler_ema` tốt nhất theo macro-F1 `0.5679`, checkpoint được chọn là `lws`.
- Trong hybrid tail ablation, clean đạt macro-F1 `0.5470`, còn hybrid + tail10 safe đạt `0.5437`.
- Trong single-encoder ablation mới, tốt nhất hiện tại là `singleenc_pool_meta_sampler05` từ repo `report_single_encoder_no_freeze` với macro-F1 `0.4898`; tất cả single-encoder runs hiện vẫn thấp hơn controlled dual-image baselines.
- Với recipe clean không augmentation, thêm metadata cho EfficientNetV2-B2 tăng macro-F1 từ `0.5223` lên `0.5430` (`+0.0206`).
- Approach dùng dermoscopic masks đã được chạy trong artifact legacy `results_effnetv2_balanced_pretrained`, nhưng kết quả hiện thấp và chưa được đưa vào controlled conclusions.
- Dermoscopic-only clean chưa được đưa vào bảng vì artifact đã tải chỉ có split/data summary, chưa có `metrics.json` cho `with_metadata` và `no_metadata`.

### Bảng controlled tóm tắt

| Nhóm | Backbone | Metadata | Fusion | Loss / post | Augment / balance | Macro-F1 | Dice macro | Bal. acc | Accuracy | Artifact |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| Backbone | `convnext_base` | Không | concat | focal | sampler `0.5`, clean | 0.5518 | 0.5518 | 0.5493 | 0.7739 | `report_backbone_convnext_base_nometa_concat_focal_s05_clean` |
| Backbone | `efficientnet_b2` | Không | concat | focal | sampler `0.5`, clean | 0.5326 | 0.5326 | 0.5571 | 0.7462 | `report_backbone_effb2_nometa_concat_focal_s05_clean` |
| Backbone | `tf_efficientnetv2_b2` | Không | concat | focal | sampler `0.5`, clean | 0.5223 | 0.5223 | 0.5186 | 0.7376 | `report_backbone_effv2b2_nometa_concat_focal_s05_clean` |
| Metadata | `tf_efficientnetv2_b2` | Có | concat | focal | sampler `0.5`, clean | 0.5430 | 0.5430 | 0.5614 | 0.6813 | `report_effv2b2_meta_concat_focal_s05_clean` |
| Augmentation | `tf_efficientnetv2_b2` | Có | concat | focal | `INF +15` source-safe | 0.5828 | 0.5828 | 0.5833 | 0.7576 | `report_effv2b2_meta_concat_focal_s05_aug_inf15_safe` |
| Fusion | `tf_efficientnetv2_b2` | Có | cross-attention | focal | sampler `0.5`, clean | 0.5599 | 0.5599 | 0.5569 | 0.7615 | `report_effv2b2_meta_cross_attention_focal_s05_clean` |
| Loss | `tf_efficientnetv2_b2` | Có | concat | CE-F1 | sampler `0.5`, clean | 0.5637 | 0.5637 | 0.5725 | 0.7634 | `report_effv2b2_meta_concat_ce_f1_s05_clean` |
| Loss / Dice metric | `tf_efficientnetv2_b2` | Có | concat | CE-Dice | sampler `0.5`, clean | 0.5571 | 0.5571 | 0.5784 | 0.7309 | `report_effv2b2_meta_concat_ce_dice_s05_clean` |
| EMA/LWS | `efficientnet_b2` | Có | concat | CE + tau `0.50` + LWS | sampler `0.5`, clean | 0.5679 | 0.5679 | 0.5604 | 0.7672 | `run_tau050_sampler_ema` |
| Hybrid tail | `tf_efficientnetv2_b2` | Có | concat | focal | hybrid balance, clean | 0.5470 | 0.5470 | 0.5387 | 0.7595 | `report_effv2b2_meta_concat_focal_hybrid_clean` |
| Single-encoder | `efficientnet_b2` | Có | `shared_encoder_pool` | CE | sampler `0.5`, clean | 0.4898 | 0.4898 | 0.4566 | 0.7013 | `report_single_encoder_no_freeze/singleenc_pool_meta_sampler05` |

## 2. Controlled Ablations

### 2.1 Backbone clean

Recipe cố định: dual-image, no metadata, concat, no augmentation, focal `gamma=2.0`, weighted sampler `power=0.5`, image size `260`, seed `42`.

| Rank | Backbone | Metadata | Fusion | Loss | Balance | Macro-F1 | Bal. acc | Accuracy | ROC-AUC macro | Artifact |
|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| 1 | `convnext_base` | Không | concat | focal | sampler `0.5` | 0.5518 | 0.5493 | 0.7739 | 0.9374 | `report_backbone_convnext_base_nometa_concat_focal_s05_clean` |
| 2 | `efficientnet_b2` | Không | concat | focal | sampler `0.5` | 0.5326 | 0.5571 | 0.7462 | 0.9017 | `report_backbone_effb2_nometa_concat_focal_s05_clean` |
| 3 | `tf_efficientnetv2_b2` | Không | concat | focal | sampler `0.5` | 0.5223 | 0.5186 | 0.7376 | 0.9026 | `report_backbone_effv2b2_nometa_concat_focal_s05_clean` |

Nhận xét:

- ConvNeXt-Base là backbone mạnh nhất trong matrix clean hiện tại.
- EfficientNet-B2 cao hơn EfficientNetV2-B2 trên macro-F1 trong recipe này.
- Kết luận này chỉ áp dụng cho recipe backbone clean ở trên, không kéo sang các run legacy khác recipe.

### 2.2 Metadata ablation

Recipe cố định: EfficientNetV2-B2, concat, no augmentation, focal `gamma=2.0`, weighted sampler `power=0.5`, image size `260`, seed `42`.

| Backbone | Metadata | Fusion | Loss | Balance | Macro-F1 | Bal. acc | Accuracy | ROC-AUC macro | Artifact |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `tf_efficientnetv2_b2` | Có | concat | focal | sampler `0.5` | 0.5430 | 0.5614 | 0.6813 | 0.9197 | `report_effv2b2_meta_concat_focal_s05_clean` |
| `tf_efficientnetv2_b2` | Không | concat | focal | sampler `0.5` | 0.5223 | 0.5186 | 0.7376 | 0.9026 | `report_backbone_effv2b2_nometa_concat_focal_s05_clean` |

Nhận xét:

- Metadata tăng `+0.0206` macro-F1 và `+0.0428` balanced accuracy.
- Accuracy giảm trong cặp này, nên diễn giải nên ưu tiên macro-F1 và balanced accuracy do dữ liệu lệch lớp.

### 2.3 Augmentation source-safe

Recipe cố định: EfficientNetV2-B2, metadata concat, image concat, focal `gamma=2.0`, weighted sampler `power=0.5`, image size `260`, seed `42`.

| Rank | Backbone | Metadata | Fusion | Loss | Augmentation | Train synthetic | Val synthetic | Macro-F1 | Bal. acc | Accuracy | Artifact |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `tf_efficientnetv2_b2` | Có | concat | focal | `INF`, max/class `15` | 15 | 0 | 0.5828 | 0.5833 | 0.7576 | `report_effv2b2_meta_concat_focal_s05_aug_inf15_safe` |
| 2 | `tf_efficientnetv2_b2` | Có | concat | focal | Không augmentation | 0 | 0 | 0.5430 | 0.5614 | 0.6813 | `report_effv2b2_meta_concat_focal_s05_clean` |
| 3 | `tf_efficientnetv2_b2` | Có | concat | focal | `INF + BEN_OTH`, max/class `15` | 30 | 0 | 0.5403 | 0.5376 | 0.7681 | `report_effv2b2_meta_concat_focal_s05_aug_inf_benoth15_safe` |
| 4 | `tf_efficientnetv2_b2` | Có | concat | focal | `BEN_OTH + DF + INF + VASC`, max/class `10` | 40 | 0 | 0.5392 | 0.5484 | 0.7347 | `report_effv2b2_meta_concat_focal_s05_aug_tail10_safe` |

Nhận xét:

- Chỉ `INF-only +15` cho cải thiện rõ ràng và sạch trong matrix hiện tại.
- `INF + BEN_OTH` và `tail10` không vượt baseline clean theo macro-F1.
- Log source-safe xác nhận synthetic chỉ nằm ở train, với nguồn validation bị loại:
  - `aug_inf15_safe`: `excluded_validation_sources=8`
  - `aug_inf_benoth15_safe`: `excluded_validation_sources=10`
  - `aug_tail10_safe`: `excluded_validation_sources=54`

### 2.4 Fusion under fixed focal recipe

Recipe cố định: EfficientNetV2-B2, metadata concat, no augmentation, focal `gamma=2.0`, weighted sampler `power=0.5`, image size `260`, seed `42`.

Lưu ý: `co_attention` đang được giữ để trace artifact, nhưng không nên tính như một hướng kết luận độc lập nếu xem nó là biến thể/alias gần với `cross_attention`. Fusion matrix hiện còn thiếu `moe` và `shared_private`; chạy bổ sung bằng `run_missing_fusion_ablation.sh`.

| Rank | Backbone | Metadata | Fusion | Loss | Balance | Macro-F1 | Bal. acc | Accuracy | ROC-AUC macro | Artifact |
|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| 1 | `tf_efficientnetv2_b2` | Có | `cross_attention` | focal | sampler `0.5` | 0.5599 | 0.5569 | 0.7615 | 0.9343 | `report_effv2b2_meta_cross_attention_focal_s05_clean` |
| 2 | `tf_efficientnetv2_b2` | Có | `co_attention` | focal | sampler `0.5` | 0.5535 | 0.5590 | 0.7662 | 0.9367 | `report_effv2b2_meta_co_attention_focal_s05_clean` |
| 3 | `tf_efficientnetv2_b2` | Có | `low_rank_bilinear` | focal | sampler `0.5` | 0.5440 | 0.5438 | 0.7471 | 0.9192 | `report_effv2b2_meta_low_rank_bilinear_focal_s05_clean` |
| 4 | `tf_efficientnetv2_b2` | Có | concat | focal | sampler `0.5` | 0.5430 | 0.5614 | 0.6813 | 0.9197 | `report_effv2b2_meta_concat_focal_s05_clean` |

Nhận xét:

- `cross_attention` đứng đầu theo macro-F1 trong fusion matrix controlled.
- `co_attention` có accuracy và ROC-AUC hơi cao hơn, nhưng không nên dùng làm nhánh kết luận riêng nếu coi nó trùng/na ná `cross_attention`.
- `low_rank_bilinear` gần như ngang concat baseline.
- Cần bổ sung `moe` và `shared_private` trước khi chốt fusion matrix.

### 2.5 Loss under fixed concat recipe

Recipe cố định: EfficientNetV2-B2, metadata concat, image concat, no augmentation, weighted sampler `power=0.5`, image size `260`, seed `42`.

| Rank | Backbone | Metadata | Fusion | Loss | Balance | Macro-F1 | Dice macro | Bal. acc | Accuracy | ROC-AUC macro | Artifact |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `tf_efficientnetv2_b2` | Có | concat | `ce_f1` | sampler `0.5` | 0.5637 | 0.5637 | 0.5725 | 0.7634 | 0.9346 | `report_effv2b2_meta_concat_ce_f1_s05_clean` |
| 2 | `tf_efficientnetv2_b2` | Có | concat | `ce` | sampler `0.5` | 0.5602 | 0.5602 | 0.5666 | 0.7548 | 0.9363 | `report_effv2b2_meta_concat_ce_s05_clean` |
| 3 | `tf_efficientnetv2_b2` | Có | concat | `ce_dice` | sampler `0.5` | 0.5571 | 0.5571 | 0.5784 | 0.7309 | 0.9277 | `report_effv2b2_meta_concat_ce_dice_s05_clean` |
| 4 | `tf_efficientnetv2_b2` | Có | concat | focal | sampler `0.5` | 0.5430 | 0.5430 | 0.5614 | 0.6813 | 0.9197 | `report_effv2b2_meta_concat_focal_s05_clean` |
| 5 | `tf_efficientnetv2_b2` | Có | concat | `ldam` | sampler `0.5` | 0.5368 | 0.5368 | 0.5570 | 0.7395 | 0.9243 | `report_effv2b2_meta_concat_ldam_s05_clean` |

Nhận xét:

- `ce_f1` là loss mạnh nhất trong matrix controlled hiện tại.
- `dice_macro` hiện trùng với macro-F1 trong artifact, nên thứ hạng Dice macro giống thứ hạng macro-F1.
- `ce_dice` không đứng đầu theo Dice macro/macro-F1, nhưng có balanced accuracy cao nhất trong loss matrix (`0.5784`).
- `ce` và `ce_dice` cũng vượt focal baseline theo macro-F1/Dice macro.
- `ldam` thấp nhất trong 5 loss ở recipe này.

### 2.6 EMA/LWS matrix

Recipe cố định: EfficientNet-B2, metadata concat, image concat, CE loss, EMA enabled, LWS `5` epochs, image size `260`, seed `42`.

| Rank | Backbone | Metadata | Fusion | Loss | Tau | Sampler | Chọn | Macro-F1 | Bal. acc | Accuracy | ROC-AUC macro | Artifact |
|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|
| 1 | `efficientnet_b2` | Có | concat | CE + EMA/LWS | 0.50 | `power=0.5` | `lws` | 0.5679 | 0.5604 | 0.7672 | 0.9050 | `run_tau050_sampler_ema` |
| 2 | `efficientnet_b2` | Có | concat | CE + EMA/LWS | 0.25 | `power=0.5` | `lws` | 0.5536 | 0.5757 | 0.7538 | 0.9188 | `run_tau025_sampler_ema` |
| 3 | `efficientnet_b2` | Có | concat | CE + EMA/LWS | 0.25 | Không | `ema` | 0.5498 | 0.5610 | 0.7824 | 0.9313 | `run_tau025_nosampler_ema` |
| 4 | `efficientnet_b2` | Có | concat | CE + EMA/LWS | 0.00 | Không | `raw` | 0.5435 | 0.5405 | 0.7653 | 0.9390 | `run_baseline_ema` |

Nhận xét:

- Macro-F1 tốt nhất đến từ tau `0.50` + weighted sampler `power=0.5` + LWS.
- Baseline raw vẫn có ROC-AUC macro cao nhất, nên cải thiện macro-F1 của LWS không đi kèm cải thiện ROC-AUC.
- Matrix này dùng EfficientNet-B2 + CE, nên không nên trộn trực tiếp với các bảng EfficientNetV2-B2 focal/fusion/loss ở trên như cùng recipe.

### 2.7 Hybrid tail ablation

Recipe cố định: EfficientNetV2-B2, metadata concat, image concat, focal `gamma=2.0`, no weighted sampler, `balance_mode=hybrid`, image size `260`, seed `42`.

| Rank | Backbone | Metadata | Fusion | Loss | Balance | Augmentation | Train synthetic | Val synthetic | Macro-F1 | Bal. acc | Accuracy | ROC-AUC macro | Artifact |
|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `tf_efficientnetv2_b2` | Có | concat | focal | hybrid | Không augmentation | 0 | 0 | 0.5470 | 0.5387 | 0.7595 | 0.9200 | `report_effv2b2_meta_concat_focal_hybrid_clean` |
| 2 | `tf_efficientnetv2_b2` | Có | concat | focal | hybrid | `BEN_OTH + DF + INF + VASC`, max/class `10` | 40 | 0 | 0.5437 | 0.5569 | 0.7500 | 0.8985 | `report_effv2b2_meta_concat_focal_hybrid_aug_tail10_safe` |

Nhận xét:

- Hybrid clean nhỉnh hơn hybrid + tail10 safe theo macro-F1.
- Hybrid tail10 safe tăng balanced accuracy so với hybrid clean, nhưng thấp hơn về macro-F1, accuracy và ROC-AUC macro.
- Hybrid tail clean vẫn chưa vượt augmentation controlled tốt nhất hiện tại là `report_effv2b2_meta_concat_focal_s05_aug_inf15_safe` với macro-F1 `0.5828`.

### 2.8 Single-encoder ablation

Recipe đọc từ artifact: backbone `efficientnet_b2`, loss `ce`, image size `260`, seed `42`, không augmentation, validation sạch `1048` mẫu real và `0` synthetic. Phần này nên xem là ablation bổ sung cho hướng one-encoder / shared-encoder, không trộn trực tiếp vào kết luận dual-image controlled ở trên.

| Rank | Repo | Run | Metadata | Image fusion | Sampler | Macro-F1 | Bal. acc | Accuracy | ROC-AUC macro |
|---:|---|---|---|---|---|---:|---:|---:|---:|
| 1 | `report_single_encoder_no_freeze` | `singleenc_pool_meta_sampler05` | Có | `shared_encoder_pool` | `power=0.5` | 0.4898 | 0.4566 | 0.7013 | 0.8961 |
| 2 | `report_single_encoder_no_freeze` | `singleenc_canvas_meta_sampler05` | Có | `single_encoder_canvas` | `power=0.5` | 0.4842 | 0.5295 | 0.6718 | 0.9014 |
| 3 | `report_single_encoder` | `singleenc_canvas_meta_clean` | Có | `single_encoder_canvas` | Không | 0.4763 | 0.4447 | 0.7328 | 0.9082 |
| 4 | `report_single_encoder` | `singleenc_canvas_meta_sampler05` | Có | `single_encoder_canvas` | `power=0.5` | 0.4701 | 0.4665 | 0.6775 | 0.9047 |
| 5 | `report_single_encoder` | `singleenc_canvas_nometa_clean` | Không | `single_encoder_canvas` | Không | 0.4572 | 0.4426 | 0.7204 | 0.8749 |
| 6 | `report_single_encoder` | `singleenc_pool_meta_clean` | Có | `shared_encoder_pool` | Không | 0.4532 | 0.4243 | 0.7109 | 0.9175 |
| 7 | `report_single_encoder` | `singleenc_pool_meta_sampler05` | Có | `shared_encoder_pool` | `power=0.5` | 0.4381 | 0.4329 | 0.6641 | 0.8857 |
| 8 | `report_single_encoder` | `singleenc_pool_nometa_clean` | Không | `shared_encoder_pool` | Không | 0.3959 | 0.3795 | 0.6737 | 0.8654 |
| 9 | `report_single_encoder_no_freeze` | `singleenc_pool_nometa_clean` | Không | `shared_encoder_pool` | Không | 0.3931 | 0.3678 | 0.6918 | 0.8796 |

Nhận xét:

- Best single-encoder run hiện tại là `singleenc_pool_meta_sampler05` trong repo `report_single_encoder_no_freeze`, nhưng macro-F1 `0.4898` vẫn thấp hơn rõ rệt so với dual-image controlled baselines.
- Metadata vẫn giúp trong nhánh single-encoder: `singleenc_canvas_meta_clean` cao hơn `singleenc_canvas_nometa_clean` (`0.4763` vs `0.4572`), và `singleenc_pool_meta_clean` cao hơn `singleenc_pool_nometa_clean` (`0.4532` vs `0.3959`).
- Weighted sampler `power=0.5` không cho tín hiệu ổn định trong repo `report_single_encoder`: cả canvas và pool đều thấp hơn bản clean tương ứng.
- Trong cặp sampler của repo `report_single_encoder_no_freeze`, `shared_encoder_pool` dẫn đầu macro-F1, còn `single_encoder_canvas` lại có balanced accuracy cao hơn (`0.5295` vs `0.4566`).

## 3. Exploratory / Legacy Ranking

> Cảnh báo: phần này chỉ để tra artifact và so bức tranh rộng. Các run dưới đây trộn recipe, backbone backend, image size, loss, balancing, augmentation policy và trong một số trường hợp có leakage hoặc confound đã nêu trong `REPORT_AUDIT_RERUN_PLAN.md`. Không dùng chúng làm kết luận chính thay cho controlled reruns.

### Legacy top runs ngoài controlled set

| Rank cũ | Backbone / approach | Metadata | Fusion | Loss | Augment / balance | Macro-F1 | Bal. acc | Accuracy | Artifact |
|---:|---|---|---|---|---|---:|---:|---:|---|
| 3 | EffNetV2 adaptive gate | Có | adaptive gate | focal | balanced legacy | 0.5767 | 0.5918 | 0.7662 | `effnetv2_adaptive_gate_balanced_pretrained` |
| 4 | EfficientNet-B2 | Có | concat | CE-Dice | no augmentation | 0.5705 | 0.5667 | 0.7958 | `milk10k_effb2_concat_no_aug_ce_dice_no_calib` |
| 5 | EfficientNet-B2 | Có | concat | CE-F1 | no augmentation | 0.5705 | 0.5667 | 0.7958 | `milk10k_effb2_concat_no_aug_ce_f1_no_calib` |
| 6 | EfficientNet-B2 | Có | cross-attention | CE-Dice | `INF + BEN_OTH +15` legacy | 0.5688 | 0.5714 | 0.7805 | `milk10k_effb2_cross_attention_aug_inf_benoth15_ce_dice_no_calib` |
| 7 | ConvNeXt-Base | Có | concat | legacy | legacy balance | 0.5673 | 0.5967 | 0.7510 | `milk10k_dual_effb2_metadata` |
| 8 | EfficientNet-B2 | Có | co-attention | CE-Dice | no augmentation | 0.5650 | 0.5562 | 0.7910 | `milk10k_effb2_co_attention_no_aug_ce_dice_no_calib` |
| 9 | EffNetV2 adaptive gate | Có | adaptive gate | legacy | balanced `1.5` head | 0.5616 | 0.5345 | 0.7691 | `effnetv2_adaptive_gate_balanced_1point5head_pretrained` |
| 10 | EfficientNet-B2 | Có | cross-attention | CE | `INF + BEN_OTH +15` legacy | 0.5616 | 0.5562 | 0.7739 | `milk10k_effb2_cross_attention_aug_inf_benoth15_ce_no_calib` |

### Các cảnh báo legacy cần giữ

- Hai run `results_dermo_effnetv2_b2/{with_metadata,no_metadata}` không còn được gọi là best overall vì validation chứa synthetic và có source overlap train/val.
- Bảng backbone cũ bị confound vì loss, sampler, image size, checkpoint và metadata không cố định.
- Bảng loss cũ bị confound vì CE từng dùng class weight trong khi CE+Dice và CE+F1 không giữ recipe hoàn toàn giống nhau.
- Bảng fusion cũ bị confound vì concat baseline không dùng cùng balancing config với attention runs.
- `milk10k_dual_effb2_metadata` phải đọc theo `run_config.json`, không suy tên backbone từ thư mục.

### Legacy mask-based runs

> Các run này dùng `dermoscopic_mask_dir=/marimo/milk10k_train_masks` với `min_dermoscopic_mask_ratio=0.01`. Chúng cho thấy approach dùng mask đã được chạy rồi, nhưng không nên trộn vào kết luận chính vì recipe khác controlled reruns hiện tại.

| Approach | Backbone | Metadata | Loss | Balance | Macro-F1 | Bal. acc | Accuracy | Artifact |
|---|---|---|---|---|---:|---:|---:|---|
| Dermoscopic masks | `tf_efficientnetv2_b2` | Không | CE | none | 0.4455 | 0.4780 | 0.6536 | `effnetv2_masks_simple_nometadata` |
| Dermoscopic masks | `tf_efficientnetv2_b2` | Có | CE | hybrid / CE-weighted legacy | 0.3880 | 0.3841 | 0.6670 | `effnetv2_masks_balanced_cew` |
| Dermoscopic masks | `tf_efficientnetv2_b2` | Có | CE | none | Chưa có | Chưa có | Chưa có | `effnetv2_masks` |

Nhận xét:

- Approach mask đã có artifact local đầy đủ về config/audit, nên không phải hướng chưa thử.
- Hai run có metrics đều thấp hơn rõ rệt so với controlled baseline hiện tại.
- Audit mask cho thấy train có `4177` mask valid và `15` too-small; val có `1043` valid và `5` too-small.

## 4. Missing or Incomplete Artifacts

| Artifact | Trạng thái | Ghi chú |
|---|---|---|
| `report_derm_effv2b2_focal_s05_clean_ablation/no_metadata` | Chỉ có split + data summary | Chưa có `metrics.json` trong artifact đã tải |
| `report_derm_effv2b2_focal_s05_clean_ablation/with_metadata` | Không thấy trong artifact tải về | Chưa thể đưa vào bảng kết quả |
| `results_effnetv2_balanced_pretrained/effnetv2_masks` | Chỉ có split/config/audit | Chưa có `metrics.json` trong artifact local hiện tại |

Nhận xét:

- Dermoscopic-only clean split đã tồn tại và sạch (`1048` val real, `0` synthetic), nhưng chưa đủ artifact để báo cáo metric.
- Khi có `metrics.json` cho hai run này, nên thêm thành một bảng riêng và tuyệt đối không trộn với dual-image controlled matrix.

## 5. Validation / Split Integrity Notes

- Validation checksum chung của các rerun backbone, metadata, augmentation, fusion và loss là `ad42f5eded8c32831899f28ddc8f56827c77eba05c08e953b31848206ac2b645`.
- Train checksum chung của các clean reference rerun là `b242fa4dfee8cd1befedc0c0baf59f1b1b72c8dd3ec3b7ed8a1b076fe431a915`.
- Các run augmentation source-safe đều có:
  - validation `1048` real, `0` synthetic,
  - synthetic chỉ append vào train,
  - log ghi rõ số source thuộc validation bị loại trước khi append synthetic.
- Các run controlled trong báo cáo này đều lấy metric trực tiếp từ `metrics.json`, không suy diễn từ log hay tên thư mục.

## 6. Kết luận ngắn

- Nếu chỉ lấy một bảng chính để trình bày hiện tại, nên dùng các bảng controlled ở Mục 2 thay vì bảng xếp hạng hỗn hợp cũ.
- Trong controlled reruns hiện tại:
  - backbone tốt nhất là `convnext_base`,
  - metadata giúp EfficientNetV2-B2,
  - augmentation tốt nhất là `INF-only +15` source-safe,
  - fusion tốt nhất là `cross_attention`,
  - loss tốt nhất là `ce_f1`,
  - Dice macro hiện trùng macro-F1 trong `metrics.json`; theo metric này `ce_f1` vẫn đứng đầu loss matrix, còn `ce_dice` đáng giữ vì balanced accuracy cao nhất,
  - EMA/LWS tốt nhất là `run_tau050_sampler_ema` với checkpoint `lws`,
  - hybrid tail clean không vượt augmentation controlled tốt nhất `report_effv2b2_meta_concat_focal_s05_aug_inf15_safe`, và hybrid tail10 safe thấp hơn hybrid clean theo macro-F1,
  - các run single-encoder mới đã được tải và cho thấy best one-encoder result hiện là `singleenc_pool_meta_sampler05` (`0.4898` macro-F1), vẫn thấp hơn đáng kể so với các dual-image controlled baselines,
  - approach dùng dermoscopic masks đã được chạy rồi trong legacy artifacts, nhưng các run có metrics hiện thấp hơn đáng kể so với controlled baseline.
- Phần exploratory / legacy vẫn hữu ích để tra những gì đã thử, nhưng không còn là nền cho kết luận chính.
