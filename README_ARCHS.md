# MILK10k Lightweight Dual-Encoder Architectures

README này ghi cách chạy các architecture dual-encoder mới và tóm tắt ý tưởng của từng model. Các script dùng paired clinical + dermoscopic images của MILK10k, ưu tiên classification performance, còn retrieval metrics dùng để kiểm tra chất lượng alignment.

## 1. Chạy benchmark 5 epoch cho tất cả architecture

Chạy script này trên máy GPU của bạn, không chạy trong sandbox:

```bash
./run_milk10k_5epoch_architecture_benchmark.sh
```

Mặc định:

- `DATA_DIR=.`: thư mục hiện tại chứa `MILK10k_Training_GroundTruth.csv`, `MILK10k_Training_Metadata.csv`, `MILK10k_Training_Input/`
- `OUTPUT_DIR=dual_encoder_5epoch_benchmark`
- `MODEL=efficientnet_b0`
- `BATCH_SIZE=32`
- `NUM_WORKERS=4`
- `SEED=42`
- training flags: `--epochs 5 --class-weight`
- AMP mặc định tắt trong benchmark. Bật nếu cần bằng `USE_AMP=1`.

Override nếu cần:

```bash
DATA_DIR=/path/to/MILK10k \
OUTPUT_DIR=runs_5epoch_effb0 \
MODEL=efficientnet_b0 \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
USE_AMP=0 \
./run_milk10k_5epoch_architecture_benchmark.sh
```

CLIP và SigLIP trong benchmark dùng tổng 5 epoch: `--pretrain-epochs 2 --finetune-epochs 3`.

## 2. Chạy từng architecture riêng

Ví dụ chạy fusion baseline:

```bash
python train_milk10k_fusion_dual_encoder_v2.py \
  --data-dir . \
  --output-dir dual_encoder_single_runs \
  --model efficientnet_b0 \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 4 \
  --seed 42 \
  --class-weight
```

Ví dụ chạy partial channel attention:

```bash
python train_milk10k_partial_channel_attention_dual_encoder.py \
  --data-dir . \
  --output-dir dual_encoder_single_runs \
  --model efficientnet_b0 \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 4 \
  --seed 42 \
  --class-weight
```

Ví dụ chạy CLIP-style 2-stage:

```bash
python train_milk10k_clip_dual_encoder.py \
  --data-dir . \
  --output-dir dual_encoder_single_runs \
  --model efficientnet_b0 \
  --epochs 5 \
  --pretrain-epochs 2 \
  --finetune-epochs 3 \
  --batch-size 32 \
  --num-workers 4 \
  --seed 42 \
  --class-weight
```

Backbone lightweight hiện có:

```text
efficientnet_b0, mobilenetv2, convnext_tiny
```

`resnet50` cũng có trong config để scale nhẹ sau khi model tốt nhất rõ ràng hơn.

## 3. Output của mỗi run

Mỗi architecture sẽ ghi vào:

```text
<output-dir>/<architecture>_<model>/
```

Các file chính:

- `splits/train.csv`, `splits/val.csv`: split lesion-level cố định theo seed
- `*_history.csv`: train/val loss, accuracy theo epoch
- `*_best.pt`: checkpoint tốt nhất theo validation loss
- `*_metrics.json`: classification metrics
- `*_retrieval_metrics.json`: clinical-to-derm và derm-to-clinical retrieval metrics
- `*_confusion_matrix.csv`: confusion matrix
- `*_per_class_metrics.csv`: per-class precision/recall/specificity/AUC
- `*_val_predictions.csv`: prediction chi tiết trên validation set
- `*_run_config.json`: args và architecture config

Metric nên ưu tiên đọc:

1. `balanced_accuracy`
2. `f1_macro`
3. `roc_auc_macro_ovr`
4. rare-class recall/sensitivity trong `*_per_class_metrics.csv`
5. retrieval `recall_at_1`, `recall_at_5`, `mrr`

## 4. Architecture idea log

### `fusion_v2`

File: `train_milk10k_fusion_dual_encoder_v2.py`

Supervised dual encoder cơ bản. Clinical image đi qua clinical encoder, dermoscopic image đi qua dermoscopic encoder. Fusion dùng:

```text
[clinical_features, dermoscopic_features, abs(diff), product]
```

Đây là baseline chính để so mọi architecture khác.

### `multitask`

File: `train_milk10k_multitask_dual_encoder.py`

Train classification và contrastive alignment cùng lúc. Loss:

```text
classification_loss + contrastive_weight * clinical_derm_InfoNCE
```

Ý tưởng: ép embedding hai modality align tốt hơn nhưng vẫn optimize trực tiếp cho diagnosis.

### `clip`

File: `train_milk10k_clip_dual_encoder.py`

Two-stage CLIP-style:

1. pretrain bằng symmetric InfoNCE clinical↔dermoscopy
2. fine-tune bằng supervised fusion classifier

Phù hợp để kiểm tra liệu alignment trước có giúp classification không.

### `siglip`

File: `train_milk10k_siglip_dual_encoder.py`

Giống CLIP-style nhưng dùng pairwise sigmoid loss thay vì softmax InfoNCE. Ý tưởng là giảm phụ thuộc vào batch normalization kiểu contrastive full-matrix, hữu ích nếu CLIP loss không ổn định.

### `sm3`

File: `train_milk10k_sm3_dual_encoder.py`

SM3-inspired: classification + contrastive alignment + auxiliary MONET attribute prediction. Dùng các cột MONET trong metadata để tạo supervision phụ.

Ý tưởng: model không chỉ học diagnosis label mà còn học visual attributes liên quan tới lesion.

### `siamese`

File: `train_milk10k_siamese_shared_encoder.py`

Control model dùng shared encoder cho cả clinical và dermoscopic image. Nếu yếu hơn untied encoder thì chứng minh modality gap đủ lớn và cần encoder riêng.

### `late_fusion`

File: `train_milk10k_late_fusion_ensemble.py`

Clinical branch và dermoscopic branch tạo logits riêng, sau đó average logits. Đây là baseline thực dụng: không fusion feature phức tạp, chỉ kiểm tra hai modality cộng vote có mạnh không.

### `mil_attention`

File: `train_milk10k_mil_attention_dual_encoder.py`

Xem clinical và dermoscopic image như 2 instances trong một lesion bag. Attention pooling học view nào quan trọng hơn cho từng sample.

Phù hợp nếu một số lesion clinical rõ hơn, một số lesion dermoscopy rõ hơn.

### `partial_channel_attention`

File: `train_milk10k_partial_channel_attention_dual_encoder.py`

YOLO-style partial idea: chỉ apply channel attention lên một phần fused vector, phần còn lại giữ nguyên. Nhẹ hơn full attention và giảm overfit.

Fusion base vẫn là concat/diff/product, nhưng nửa đầu channels được gated bằng learned attention.

### `partial_cross_attention`

File: `train_milk10k_partial_cross_attention_dual_encoder.py`

Project clinical và dermoscopic features xuống reduced dimension, chạy cross-attention trong không gian nhỏ, rồi concat output attention với fused pair features gốc.

Ý tưởng: lấy interaction giữa modality nhưng không trả giá full cross-attention trên toàn feature dimension.

### `metadata_fusion`

File: `train_milk10k_metadata_fusion_dual_encoder.py`

Image dual encoder + metadata tabular như age, sex, site, skin tone. Đây là ablation image+tabular, không còn thuần image-only.

Nên đọc riêng vì nó có thể tăng score nhờ metadata chứ không nhất thiết do visual architecture tốt hơn.

### `cross_attention`

File: `train_milk10k_cross_attention_dual_encoder.py`

Full lightweight cross-attention trên clinical/dermoscopic features. Đây là attention ablation nặng hơn partial variants.

Nên so với `partial_channel_attention` và `partial_cross_attention`; nếu full cross-attention không hơn rõ ràng thì không nên dùng làm default.

## 5. Thứ tự đọc kết quả đề xuất

Sau khi chạy benchmark, so theo thứ tự:

1. `fusion_v2` vs `train_milk10k_dual_encoder.py` cũ
2. `multitask` vs `fusion_v2`
3. `partial_channel_attention`, `partial_cross_attention`, `mil_attention` vs `fusion_v2`
4. `clip`, `siglip` vs `multitask`
5. `sm3` nếu MONET auxiliary giúp macro-F1 hoặc rare-class recall
6. `metadata_fusion` đọc riêng vì có tabular metadata
7. chỉ scale backbone nếu architecture lightweight thắng rõ

## 6. Quick syntax check

Không train model, chỉ kiểm tra import/syntax:

```bash
python -m py_compile milk10k_dual_encoder_common.py milk10k_dual_encoder/*.py train_milk10k_*.py
```
