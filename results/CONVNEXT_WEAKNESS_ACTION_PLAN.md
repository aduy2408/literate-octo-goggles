# ConvNeXt Weakness Analysis And Next Runs

## Tóm tắt

ConvNeXt-Base trong controlled backbone clean hiện tại không phải là "hỏng toàn diện": accuracy `0.7739` và ROC-AUC macro `0.9374` vẫn cao. Nhưng nếu mục tiêu là macro-F1 trên dữ liệu lệch lớp, kết quả này chưa đủ tốt vì tail classes bị bỏ rơi rõ rệt.

Run chính được phân tích:

| Run | Macro-F1 | Bal. acc | Accuracy | ROC-AUC macro | Recipe |
|---|---:|---:|---:|---:|---|
| `report_backbone_convnext_base_nometa_concat_focal_s05_clean` | 0.5518 | 0.5493 | 0.7739 | 0.9374 | ConvNeXt-Base, no metadata, focal, sampler `0.5`, clean |
| `convnext_base_no_meta_original_only` | 0.5318 | 0.5712 | 0.7147 | 0.8854 | Legacy / old recipe |

Diễn giải ngắn: ConvNeXt controlled đứng đầu trong bảng backbone clean, nhưng đó là so trong một recipe hẹp. Nó vẫn yếu ở tail recall và các cặp lớp dễ nhầm, nên chưa nên xem là backbone ổn để chốt.

## Yếu điểm chính

### 1. Tail classes collapse

Per-class của controlled ConvNeXt cho thấy các lớp rất ít mẫu đang bị dự đoán gần như không ra:

| Class | Support val | Precision | Recall | F1 | Vấn đề |
|---|---:|---:|---:|---:|---|
| `BEN_OTH` | 9 | 0.0000 | 0.0000 | 0.0000 | Không bắt được mẫu nào |
| `MAL_OTH` | 2 | 0.0000 | 0.0000 | 0.0000 | Không bắt được mẫu nào |
| `INF` | 10 | 1.0000 | 0.2000 | 0.3333 | Quá conservative, bỏ sót 8/10 |
| `SCCKA` | 95 | 0.5596 | 0.6421 | 0.5980 | Nhầm nhiều với BCC/BKL/AKIEC |
| `BKL` | 109 | 0.5849 | 0.5688 | 0.5767 | Nhầm nhiều với BCC/SCCKA |

Đây là lý do macro-F1 thấp: các lớp head như `BCC`, `NV`, `VASC`, `DF` kéo metric tổng lên, nhưng macro-F1 phạt mạnh các lớp tail F1 bằng 0.

### 2. Confusion tập trung vào vài cụm bệnh

Các lỗi lớn trong confusion matrix:

| True class | Nhầm chính | Mức lỗi |
|---|---|---:|
| `BEN_OTH` | `MEL` | 4/9 |
| `BEN_OTH` | `BCC` | 2/9 |
| `INF` | `BCC` | 6/10 |
| `MAL_OTH` | `BCC` | 2/2 |
| `BKL` | `BCC` | 19/109 |
| `BKL` | `SCCKA` | 17/109 |
| `AKIEC` | `SCCKA` | 12/61 |
| `MEL` | `NV` | 17/90 |
| `NV` | `MEL` | 16/149 |
| `SCCKA` | `BCC` | 13/95 |
| `SCCKA` | `BKL` | 12/95 |

Điểm đáng chú ý: `INF` và `MAL_OTH` bị hút về `BCC`; `BEN_OTH` bị hút về `MEL/BCC`; `MEL/NV` vẫn là cặp nhầm lớn. Đây không chỉ là vấn đề backbone, mà là vấn đề decision boundary cho tail và các cụm tương đồng.

### 3. Recipe hiện tại chưa công bằng cho ConvNeXt nếu muốn chốt backbone

Controlled ConvNeXt đang chạy `no metadata`, focal `gamma=2.0`, sampler `power=0.5`, image size `260`.

Các kết quả mới trong report cho thấy recipe có ảnh hưởng lớn:

- Metadata giúp EfficientNetV2-B2 tăng macro-F1 từ `0.5223` lên `0.5430`.
- Augmentation `INF-only +15` source-safe đạt `0.5828`, cao hơn mọi run ConvNeXt hiện tại.
- EMA/LWS với EfficientNet-B2 + CE đạt `0.5679`, chọn checkpoint `lws`.
- Hybrid tail clean với EfficientNetV2-B2 chỉ `0.5470`, tức tail strategy không tự động thắng nếu recipe chưa đúng.

Vì vậy kết luận hợp lý là: ConvNeXt chưa được khai thác đúng ở metadata/tail/calibration recipe, nhưng bản no-metadata focal hiện tại không đủ tốt để dùng làm best final.

## Nên khắc phục chỗ nào

### Ưu tiên 1: thêm metadata cho ConvNeXt

ConvNeXt controlled hiện tại là no-metadata. Nếu metadata đã giúp EfficientNetV2-B2, cần chạy ConvNeXt cùng metadata trước khi loại nó.

Run nên chạy:

| Mục tiêu | Recipe đề xuất | Kỳ vọng |
|---|---|---|
| ConvNeXt metadata baseline | ConvNeXt-Base, metadata concat, image concat, focal `gamma=2.0`, sampler `0.5`, clean | Kiểm tra metadata có cứu tail/cụm nhầm không |
| ConvNeXt metadata + CE-F1 | ConvNeXt-Base, metadata concat, CE-F1, sampler `0.5`, clean | So với loss tốt nhất matrix hiện tại |
| ConvNeXt metadata + CE | ConvNeXt-Base, metadata concat, CE, sampler `0.5`, clean | Baseline cho EMA/LWS nếu cần |

Không nên chạy thêm no-metadata trước khi có metadata result, vì điểm yếu hiện tại có thể đến từ thiếu signal metadata chứ không chỉ từ backbone.

### Ưu tiên 2: xử lý tail bằng augmentation source-safe, không chỉ sampler

ConvNeXt đang fail nặng ở `BEN_OTH`, `INF`, `MAL_OTH`. Weighted sampler `0.5` không đủ.

Run nên chạy:

| Mục tiêu | Recipe đề xuất | Lý do |
|---|---|---|
| ConvNeXt + `INF-only +15` | ConvNeXt-Base, metadata concat, focal hoặc CE-F1, sampler `0.5`, source-safe INF augmentation | `INF-only +15` là augmentation tốt nhất hiện tại |
| ConvNeXt + tail10 safe | ConvNeXt-Base, metadata concat, CE-F1, sampler `0.5`, source-safe `BEN_OTH + DF + INF + VASC` | Test trực tiếp tail support với backbone này |
| ConvNeXt + targeted BEN_OTH/INF | ConvNeXt-Base, metadata concat, CE-F1, source-safe BEN_OTH + INF augmentation | Nhắm đúng hai lớp collapse thay vì augment rộng |

Không nên kỳ vọng tail10 tự thắng: hybrid tail10 safe đang thấp hơn hybrid clean. Nên xem tail10 là diagnostic, còn `INF-only +15` là candidate chính vì đã có bằng chứng tốt hơn.

### Ưu tiên 3: thử CE/CE-F1 và EMA/LWS cho ConvNeXt

Focal hiện tại đang cho ROC-AUC cao nhưng macro-F1 chưa tốt. Điều này gợi ý ranking score không tệ, nhưng threshold/decision boundary hoặc training objective chưa phù hợp.

Run nên chạy:

| Mục tiêu | Recipe đề xuất | Lý do |
|---|---|---|
| ConvNeXt CE-F1 clean | Metadata concat, CE-F1, sampler `0.5` | CE-F1 đang là loss tốt nhất trong controlled loss matrix |
| ConvNeXt CE + EMA/LWS tau `0.50` | Metadata concat, CE, sampler `0.5`, EMA, LWS 5 epochs, tau `0.50` | EMA/LWS tốt nhất hiện tại dùng tau `0.50` + sampler + LWS |
| ConvNeXt CE + EMA/LWS tau `0.25` | Metadata concat, CE, sampler `0.5`, EMA, LWS 5 epochs, tau `0.25` | Kiểm tra tau nhạy với backbone không |

Nếu chỉ đủ tài nguyên cho một nhánh EMA/LWS, chọn tau `0.50` + sampler vì đang là best trong EMA/LWS matrix.

## Ma trận chạy thêm đề xuất

### Minimal set

Chạy 4 run trước để quyết định có giữ ConvNeXt không:

| Rank ưu tiên | Run đề xuất | Recipe |
|---:|---|---|
| 1 | `convnext_meta_concat_ce_f1_s05_clean` | ConvNeXt-Base, metadata concat, CE-F1, sampler `0.5`, clean |
| 2 | `convnext_meta_concat_focal_s05_aug_inf15_safe` | ConvNeXt-Base, metadata concat, focal, sampler `0.5`, `INF +15` source-safe |
| 3 | `convnext_meta_concat_ce_f1_s05_aug_inf15_safe` | ConvNeXt-Base, metadata concat, CE-F1, sampler `0.5`, `INF +15` source-safe |
| 4 | `convnext_meta_concat_ce_tau050_sampler_ema_lws` | ConvNeXt-Base, metadata concat, CE, sampler `0.5`, tau `0.50`, EMA + LWS |

Điều kiện giữ ConvNeXt làm candidate:

- Macro-F1 vượt `0.5828`, hoặc
- Macro-F1 gần `0.5828` nhưng tail F1 cải thiện rõ ở `BEN_OTH`, `INF`, `MAL_OTH`, hoặc
- Balanced accuracy vượt rõ các baseline hiện tại mà không làm accuracy sụp.

### Expanded set

Nếu minimal set có tín hiệu tốt, chạy thêm:

| Run đề xuất | Recipe |
|---|---|
| `convnext_meta_concat_ce_s05_clean` | CE clean để tách tác động CE-F1 |
| `convnext_meta_concat_ce_tau025_sampler_ema_lws` | EMA/LWS tau `0.25` |
| `convnext_meta_concat_ce_f1_s05_aug_tail10_safe` | Tail10 safe với CE-F1 |
| `convnext_meta_cross_attention_ce_f1_s05_clean` | Kiểm tra fusion tốt hơn concat không |

## Tiêu chí đọc kết quả lần tới

Không chỉ nhìn macro-F1 tổng. Với ConvNeXt, cần đọc thêm:

- `BEN_OTH`, `INF`, `MAL_OTH` F1/recall: nếu vẫn bằng 0 hoặc gần 0 thì recipe chưa ổn.
- `MEL` vs `NV`: giảm nhầm hai chiều mà không hạ recall của cả hai.
- `BKL`/`SCCKA`/`AKIEC`: giảm nhầm qua lại, đặc biệt `AKIEC -> SCCKA` và `BKL -> BCC/SCCKA`.
- ROC-AUC macro vs macro-F1: nếu AUC cao nhưng F1 thấp, ưu tiên calibration/logit adjustment/LWS thay vì đổi backbone ngay.

## Kết luận

ConvNeXt hiện tại không đáng chốt làm best final vì tail collapse quá rõ. Nhưng cũng chưa nên loại hẳn, vì run đang là no-metadata focal clean và chưa được thử với các recipe đang thắng ở matrix khác.

Thứ tự hợp lý là:

1. Chạy ConvNeXt + metadata + CE-F1 clean.
2. Chạy ConvNeXt + metadata + `INF-only +15` source-safe.
3. Chạy ConvNeXt + CE + EMA/LWS tau `0.50` nếu CE baseline có tín hiệu.
4. Chỉ mở rộng sang cross-attention/tail10 nếu 3 bước trên cải thiện tail classes thật sự.
