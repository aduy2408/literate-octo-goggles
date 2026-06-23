# Current Model And Approach Conclusions

## Kết luận tổng hợp

| Model / hướng | Best evidence hiện có | Điểm mạnh | Điểm yếu chính | Kết luận hiện tại | Nên làm tiếp |
|---|---|---|---|---|---|
| `ConvNeXt-Base` | Backbone clean no-metadata: macro-F1 `0.5518`, bal. acc `0.5493`, acc `0.7739`, ROC-AUC `0.9374` | Tốt nhất trong 3 backbone clean; accuracy và ROC-AUC cao | Tail collapse: `BEN_OTH` F1 `0`, `MAL_OTH` F1 `0`, `INF` recall `0.2`; chưa có metadata/loss/augment/EMA matrix riêng | Chưa nên chốt làm final, nhưng cũng chưa nên loại vì mới chỉ có no-metadata focal clean | Chạy ConvNeXt + metadata + CE-F1 clean, rồi ConvNeXt + `INF-only +15` source-safe |
| `EfficientNet-B2` | Backbone clean no-metadata: macro-F1 `0.5326`; EMA/LWS best `run_tau050_sampler_ema`: macro-F1 `0.5679`, chọn `lws` | EMA/LWS + CE có tín hiệu tốt; tau `0.50` + sampler + LWS vượt CE/F1 legacy gần đây | Backbone clean thấp hơn ConvNeXt; EMA/LWS recipe khác matrix controlled chính nên chưa so trực tiếp được với EfficientNetV2-B2 focal/augment | Candidate tốt cho nhánh CE + EMA/LWS, nhất là khi cần cải thiện macro-F1 mà không đổi quá nhiều kiến trúc | Mở rộng EMA/LWS trên metadata + augmentation source-safe; nếu có compute, test CE-F1 + metadata cho EffB2 |
| `EfficientNetV2-B2` | Best controlled augmentation: `report_effv2b2_meta_concat_focal_s05_aug_inf15_safe` macro-F1 `0.5828`; best loss clean `ce_f1` macro-F1 `0.5637`; CE-Dice clean có bal. acc tốt nhất `0.5784`; best fusion clean `cross_attention` macro-F1 `0.5599` | Có evidence đầy đủ nhất: metadata, augmentation, fusion, loss, hybrid tail; best current result thuộc model này | Backbone clean no-metadata yếu nhất trong 3 (`0.5223`); cần recipe tốt mới mạnh; hybrid tail không vượt augmentation tốt nhất; CE-Dice vẫn không cứu `BEN_OTH`/`MAL_OTH` | Đây là model nên giữ làm baseline chính hiện tại vì đã có best controlled result `0.5828` | Ưu tiên kết hợp các tín hiệu tốt: metadata + `INF-only +15` + CE-F1; thêm CE-Dice + augmentation để kiểm tra balanced accuracy |

## Bảng theo approach

| Approach | Best run hiện tại | Macro-F1 | Kết luận | Hành động tiếp theo |
|---|---|---:|---|---|
| Backbone clean | `report_backbone_convnext_base_nometa_concat_focal_s05_clean` | 0.5518 | ConvNeXt đứng đầu khi cố định no-metadata + focal + sampler `0.5`, nhưng tail rất yếu | Không chốt chỉ dựa vào backbone clean; phải chạy thêm metadata/loss/augment cho ConvNeXt |
| Metadata | `report_effv2b2_meta_concat_focal_s05_clean` vs no-meta EffV2-B2 | 0.5430 vs 0.5223 | Metadata giúp EffV2-B2 `+0.0206` macro-F1 và `+0.0428` bal. acc | Áp dụng metadata cho ConvNeXt và EffB2 trước khi loại/chọn backbone |
| Augmentation | `report_effv2b2_meta_concat_focal_s05_aug_inf15_safe` | 0.5828 | `INF-only +15` source-safe là cải thiện rõ nhất và là best current result | Chạy lại `INF-only +15` trên ConvNeXt metadata và thử kết hợp với CE-F1 |
| Fusion | `report_effv2b2_meta_cross_attention_focal_s05_clean` | 0.5599 | Cross-attention tốt nhất fusion matrix clean, nhỉnh hơn concat focal clean `0.5430` | Chỉ mở rộng cross-attention sau khi cố định loss/augment tốt; không ưu tiên hơn augmentation |
| Loss | `report_effv2b2_meta_concat_ce_f1_s05_clean` | 0.5637 | CE-F1 là loss clean tốt nhất theo macro-F1, vượt focal clean `0.5430`; CE-Dice đứng sau với macro-F1 `0.5571` nhưng có balanced accuracy clean cao nhất `0.5784` | Test CE-F1 với augmentation `INF-only +15`; test thêm CE-Dice + `INF-only +15` nếu muốn ưu tiên balanced accuracy |
| EMA/LWS | `run_tau050_sampler_ema` | 0.5679 | CE + tau `0.50` + sampler + LWS tốt nhất trong EMA/LWS matrix | Mang EMA/LWS sang ConvNeXt hoặc EffV2-B2 sau khi có CE baseline tương ứng |
| Hybrid tail | `report_effv2b2_meta_concat_focal_hybrid_clean` | 0.5470 | Hybrid clean hơn hybrid tail10 safe `0.5437`, nhưng đều chưa vượt augmentation `INF-only +15` | Không ưu tiên hybrid tail làm hướng chính; chỉ dùng như diagnostic cho tail balancing |

## Ranking thực dụng hiện tại

| Rank | Candidate | Vì sao | Rủi ro |
|---:|---|---|---|
| 1 | EffV2-B2 + metadata + concat + focal + sampler `0.5` + `INF-only +15` source-safe | Best controlled macro-F1 `0.5828` | Có thể chưa tối ưu vì loss vẫn là focal, chưa kết hợp CE-F1 |
| 2 | EffB2 + CE + tau `0.50` + sampler `0.5` + EMA/LWS | Macro-F1 `0.5679`, tốt nhất EMA/LWS | Recipe khác, chưa biết khi thêm augmentation/metadata matrix đầy đủ sẽ ra sao |
| 3 | EffV2-B2 + metadata + concat + CE-F1 + sampler `0.5` clean | Best loss clean `0.5637` | Chưa có augmentation đi kèm |
| 4 | EffV2-B2 + metadata + concat + CE-Dice + sampler `0.5` clean | Macro-F1 `0.5571`, balanced accuracy `0.5784` cao nhất trong clean loss matrix | Không cứu được `MAL_OTH`, `BEN_OTH` vẫn thấp |
| 5 | EffV2-B2 + metadata + cross-attention + focal clean | Best fusion clean `0.5599` | Fusion chưa thắng augmentation/loss tốt nhất |
| 6 | ConvNeXt-Base clean no-metadata focal | Best backbone clean `0.5518` | Tail collapse, thiếu metadata/loss/augment evidence |

## Ma trận nên chạy tiếp

| Ưu tiên | Run đề xuất | Mục tiêu |
|---:|---|---|
| 1 | `effv2b2_meta_concat_ce_f1_s05_aug_inf15_safe` | Kiểm tra kết hợp tốt nhất hiện có: CE-F1 clean + `INF-only +15` augmentation |
| 2 | `effv2b2_meta_concat_ce_dice_s05_aug_inf15_safe` | Kiểm tra CE-Dice vì clean balanced accuracy đang cao nhất |
| 3 | `convnext_meta_concat_ce_f1_s05_clean` | Xem ConvNeXt có còn yếu khi thêm metadata và loss tốt nhất không |
| 4 | `convnext_meta_concat_focal_s05_aug_inf15_safe` | Xem augmentation tốt nhất hiện tại có cứu ConvNeXt tail không |
| 5 | `convnext_meta_concat_ce_f1_s05_aug_inf15_safe` | Candidate ConvNeXt mạnh nhất theo các tín hiệu hiện có |
| 6 | `effv2b2_meta_cross_attention_ce_f1_s05_clean` | Tách tác động fusion + loss tốt nhất |
| 7 | `effv2b2_meta_concat_ce_tau050_sampler_ema_lws` | Đưa EMA/LWS tau `0.50` sang EffV2-B2 để so với EffB2 |

## Chốt hiện tại

Nếu cần chọn model/approach ngay bây giờ:

| Mục đích | Chọn | Lý do |
|---|---|---|
| Best current controlled result | EffV2-B2 + metadata + concat + focal + sampler `0.5` + `INF-only +15` | Macro-F1 cao nhất `0.5828` |
| Hướng clean không augmentation tốt nhất | EffV2-B2 + metadata + concat + CE-F1 | Macro-F1 `0.5637`, loss matrix tốt nhất |
| Hướng clean nếu ưu tiên balanced accuracy | EffV2-B2 + metadata + concat + CE-Dice | Balanced accuracy `0.5784`, cao nhất clean loss matrix |
| Hướng calibration/post-training đáng thử | EffB2 + CE + tau `0.50` + sampler + EMA/LWS | Macro-F1 `0.5679`, LWS thắng trong matrix |
| Backbone cần điều tra thêm | ConvNeXt-Base | Clean no-metadata tốt nhưng tail fail nặng; cần metadata + CE-F1 + augmentation trước khi kết luận |

Kết luận ngắn: hiện tại nên lấy EffV2-B2 + metadata + `INF-only +15` source-safe làm baseline chính, dùng CE-F1, CE-Dice và EMA/LWS làm hướng cải thiện tiếp theo, còn ConvNeXt cần rerun đúng recipe trước khi loại hoặc chọn.
